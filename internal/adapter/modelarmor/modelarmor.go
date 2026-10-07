// Package modelarmor implements domain.ArmorClient against Cloud Model
// Armor REST API.
//
// Per ADR-152 the gateway is the SOLE Cloud Model Armor caller post-cutover
// (the chora-guardrail Go service is decommissioned). Per ADR-163 +
// feedback_model_armor_regional_endpoint the SDK endpoint MUST be pinned
// regionally:
//   modelarmor.asia-southeast1.rep.googleapis.com
// A default global endpoint returns InvalidArgument: TEMPLATE_NOT_FOUND
// against regional templates — burned ~3 hours of debug 2026-05-17.
//
// We use raw HTTP + JSON (rather than cloud.google.com/go/modelarmor) for
// the same reasons as the vendor adapters: testability with
// httptest.NewServer + injectable transport, and the gateway is the only
// caller so no SDK-abstraction benefit.
package modelarmor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// TokenProvider mirrors the gemini adapter's port — short-lived OAuth2
// access token from a WIF source in production, stub in tests.
type TokenProvider interface {
	Token(ctx context.Context) (string, error)
}

// Client implements domain.ArmorClient via Cloud Model Armor REST.
type Client struct {
	httpClient *http.Client
	tokens     TokenProvider
	endpoint   string // optional override; production builds from Location
	location   string
}

// Config groups construction inputs.
type Config struct {
	HTTPClient *http.Client
	Tokens     TokenProvider // required
	Location   string        // required — asia-southeast1 post-Phase-0.3
	Endpoint   string        // optional override (test injection)
}

// New constructs a Client. Required-field validation is fail-loud.
func New(cfg Config) (*Client, error) {
	if cfg.Tokens == nil {
		return nil, fmt.Errorf("modelarmor: TokenProvider required")
	}
	if cfg.Location == "" {
		return nil, fmt.Errorf("modelarmor: Location required (e.g. asia-southeast1)")
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 30 * time.Second}
	}
	return &Client{httpClient: hc, tokens: cfg.Tokens, endpoint: cfg.Endpoint, location: cfg.Location}, nil
}

// SanitizeUserPrompt — implements domain.ArmorClient PRE method.
func (c *Client) SanitizeUserPrompt(ctx context.Context, template, prompt string) (domain.ArmorVerdict, string, error) {
	return c.invoke(ctx, template, prompt, true)
}

// SanitizeModelResponse — implements domain.ArmorClient POST method.
func (c *Client) SanitizeModelResponse(ctx context.Context, template, response string) (domain.ArmorVerdict, string, error) {
	return c.invoke(ctx, template, response, false)
}

// invoke routes both PRE + POST through the same HTTP-call body builder.
// `isUserPrompt=true` selects sanitizeUserPrompt + the userPromptData wrap.
// `isUserPrompt=false` selects sanitizeModelResponse + modelResponseData.
func (c *Client) invoke(ctx context.Context, template, text string, isUserPrompt bool) (domain.ArmorVerdict, string, error) {
	if template == "" {
		return 0, "", fmt.Errorf("modelarmor: template required (resource name format projects/.../locations/.../templates/...)")
	}

	host := c.endpoint
	if host == "" {
		host = fmt.Sprintf("https://modelarmor.%s.rep.googleapis.com", c.location)
	}
	suffix := ":sanitizeModelResponse"
	bodyBytes, err := json.Marshal(map[string]any{"modelResponseData": dataBlock{Text: text}})
	if isUserPrompt {
		suffix = ":sanitizeUserPrompt"
		bodyBytes, err = json.Marshal(map[string]any{"userPromptData": dataBlock{Text: text}})
	}
	if err != nil {
		return 0, "", fmt.Errorf("modelarmor: marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/%s%s", host, template, suffix)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(bodyBytes))
	if err != nil {
		return 0, "", fmt.Errorf("modelarmor: build request: %w", err)
	}
	token, err := c.tokens.Token(ctx)
	if err != nil {
		return 0, "", fmt.Errorf("modelarmor: token: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+token)
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return 0, "", fmt.Errorf("modelarmor: HTTP do: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		return 0, "", fmt.Errorf("modelarmor: HTTP %d: %s", httpResp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var parsed armorResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return 0, "", fmt.Errorf("modelarmor: unmarshal response: %w", err)
	}
	return parsed.toDomain(text)
}

// ---------------------------------------------------------------------------
// Wire shapes — minimal subset of Cloud Model Armor sanitize* responses.
//
// Format follows the GA API (as of 2026-05). Top-level result wraps the
// sanitization decision + the (possibly redacted) text.
// ---------------------------------------------------------------------------

type dataBlock struct {
	Text string `json:"text"`
}

type armorResponse struct {
	SanitizationResult struct {
		FilterMatchState string `json:"filterMatchState"`
		// Per-filter sub-result blob; we don't parse individual filters
		// in Phase 2.3 — we just need the top-level verdict + redacted text.
	} `json:"sanitizationResult"`
	// Some Model Armor templates emit a `sanitizedText` field when their
	// enforcement mode is SANITIZE rather than BLOCK. We accept either.
	SanitizedText string `json:"sanitizedText"`
}

func (r armorResponse) toDomain(originalText string) (domain.ArmorVerdict, string, error) {
	out := r.SanitizedText
	if out == "" {
		out = originalText
	}
	switch r.SanitizationResult.FilterMatchState {
	case "MATCH_FOUND":
		// Per Cloud Model Armor docs, MATCH_FOUND + enforcement INSPECT_AND_BLOCK
		// means the call should be refused. We surface BLOCK verdict; the
		// service layer's IsTerminalBlock() short-circuits.
		return domain.ArmorVerdictBlock, out, nil
	case "NO_MATCH_FOUND", "":
		return domain.ArmorVerdictAllow, out, nil
	default:
		// Unknown match state — degrade safely to BLOCK rather than ALLOW.
		return domain.ArmorVerdictError, out, nil
	}
}
