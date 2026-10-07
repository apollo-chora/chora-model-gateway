// Package secrets implements domain.SecretClient against Google Secret
// Manager REST API.
//
// Naming convention per ADR-163 + secrets-and-env skill BYOA subsection:
//   BYOA per-tenant key: chora-byoa-{tenant_id}-{vendor}-api-key
//   Platform vendor key: chora-vendor-{vendor}-api-key-{env}
//
// vendor token in the secret_id is the VendorFamily string verbatim:
//   vertex_ai_gemini / vertex_ai_gemma / openai_byoa / anthropic_byoa
//
// We use raw HTTP rather than cloud.google.com/go/secretmanager for the
// same testability reasons as the other adapters (httptest.NewServer +
// injectable transport). The gateway never persists the resolved key in
// memory beyond the per-invocation call scope — caching is a Phase 4
// follow-up alongside the OTel-emitted Secret-Manager-call cost.
package secrets

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// TokenProvider — short-lived OAuth2 token, same shape as the gemini /
// modelarmor adapters.
type TokenProvider interface {
	Token(ctx context.Context) (string, error)
}

// Client implements domain.SecretClient.
type Client struct {
	httpClient  *http.Client
	tokens      TokenProvider
	project     string
	environment string // dev / staging / prod — used in the vendor-key suffix
	endpoint    string // optional override
}

// Config groups construction inputs.
type Config struct {
	HTTPClient  *http.Client
	Tokens      TokenProvider // required
	Project     string        // required — chora-489812
	Environment string        // required — dev / staging / prod
	Endpoint    string        // optional override (test injection)
}

// New constructs a Client. Required-field validation is fail-loud.
func New(cfg Config) (*Client, error) {
	if cfg.Tokens == nil {
		return nil, fmt.Errorf("secrets: TokenProvider required")
	}
	if cfg.Project == "" {
		return nil, fmt.Errorf("secrets: Project required")
	}
	if cfg.Environment == "" {
		return nil, fmt.Errorf("secrets: Environment required (dev/staging/prod)")
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 10 * time.Second}
	}
	return &Client{httpClient: hc, tokens: cfg.Tokens, project: cfg.Project, environment: cfg.Environment, endpoint: cfg.Endpoint}, nil
}

// ResolveByoaKey — implements domain.SecretClient.
//
// Returns ("", nil) when the BYOA secret does not exist (HTTP 404). Other
// errors bubble up (auth, network, permission). The service layer's
// vendor adapter MAY fall back to ResolveGlobalKey on the empty-string return.
func (c *Client) ResolveByoaKey(ctx context.Context, tenantID string, vendor domain.VendorFamily) (string, error) {
	id := fmt.Sprintf("chora-byoa-%s-%s-api-key", tenantID, vendor)
	return c.access(ctx, id)
}

// ResolveGlobalKey — implements domain.SecretClient.
func (c *Client) ResolveGlobalKey(ctx context.Context, vendor domain.VendorFamily) (string, error) {
	id := fmt.Sprintf("chora-vendor-%s-api-key-%s", vendor, c.environment)
	return c.access(ctx, id)
}

// ResolveSecret fetches an arbitrary secret by its Secret Manager ID with
// FAIL-LOUD semantics: a NOT_FOUND (404) is returned as an error, not as
// the empty string (which would let the caller silently fall through).
//
// Use for required boot-time secrets (DB DSN, signing keys, etc.) where
// missing is unrecoverable. Use ResolveByoaKey / ResolveGlobalKey instead
// for vendor-key fallback chains that tolerate missing-on-purpose.
func (c *Client) ResolveSecret(ctx context.Context, secretID string) (string, error) {
	if strings.TrimSpace(secretID) == "" {
		return "", fmt.Errorf("secrets: ResolveSecret: secretID required")
	}
	v, err := c.access(ctx, secretID)
	if err != nil {
		return "", err
	}
	if v == "" {
		return "", fmt.Errorf("secrets: required secret %q NOT_FOUND in project %s (or empty payload)", secretID, c.project)
	}
	return v, nil
}

// access fetches the latest version of a secret + returns the decoded
// payload. NOT_FOUND maps to ("", nil) so callers can chain fallbacks.
func (c *Client) access(ctx context.Context, secretID string) (string, error) {
	host := c.endpoint
	if host == "" {
		host = "https://secretmanager.googleapis.com"
	}
	url := fmt.Sprintf("%s/v1/projects/%s/secrets/%s/versions/latest:access", host, c.project, secretID)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("secrets: build request: %w", err)
	}
	token, err := c.tokens.Token(ctx)
	if err != nil {
		return "", fmt.Errorf("secrets: token: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+token)

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("secrets: HTTP do: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode == http.StatusNotFound {
		// Not an error from the gateway's perspective — caller decides
		// (BYOA absent → fall back to global; global absent → fail-loud
		// at the vendor adapter layer).
		return "", nil
	}
	if httpResp.StatusCode >= 400 {
		return "", fmt.Errorf("secrets: HTTP %d for %s: %s", httpResp.StatusCode, secretID, strings.TrimSpace(string(respBody)))
	}
	var parsed accessResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return "", fmt.Errorf("secrets: unmarshal response: %w", err)
	}
	raw, err := base64.StdEncoding.DecodeString(parsed.Payload.Data)
	if err != nil {
		return "", fmt.Errorf("secrets: base64 decode payload: %w", err)
	}
	return strings.TrimRight(string(raw), "\n"), nil
}

// ---------------------------------------------------------------------------
// Wire shapes — minimal subset of Secret Manager :access response.
// ---------------------------------------------------------------------------

type accessResponse struct {
	Name    string  `json:"name"`
	Payload payload `json:"payload"`
}

type payload struct {
	// Base64-encoded secret bytes. We decode + strip a trailing newline
	// (which `gcloud secrets versions add --data-file=` often appends).
	Data string `json:"data"`
}
