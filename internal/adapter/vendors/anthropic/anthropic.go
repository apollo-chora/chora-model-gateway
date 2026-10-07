// Package anthropic implements domain.VendorClient against the Anthropic
// /v1/messages API. BYOA-only initially (no platform-controlled global
// Anthropic key per ADR-163 §"Code consequences" — Anthropic spend is the
// tenant's responsibility on day 1).
//
// Behavioral parity with the Python reference (app/runtime.py):
//   - cache_creation_input_tokens billed at the cache-write rate, on top of
//     the discounted cache-read rate for cache_read_input_tokens;
//   - hosted web-search grounding via the web_search_20250305 server tool,
//     injected when the request's response modality is GROUNDED;
//   - citations parsed from text-block citations + search queries from
//     server_tool_use blocks;
//   - generation-config breadth (temperature, top_p, max_tokens, n, seed,
//     stop, tool_choice, response_format) with the Python runtime's
//     type-coercion rules;
//   - a connection-pooled HTTP client with a generous 300s timeout (a
//     grounded reasoning run makes several upstream calls and can take well
//     over a minute);
//   - per-registry extra headers + endpoint path overrides (relative path
//     joined onto the base URL, or an absolute URL used verbatim);
//   - ProviderError carrying the provider's own status/endpoint/body so the
//     facade can relay it instead of flattening it.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

const (
	defaultEndpoint = "https://api.anthropic.com"
	defaultVersion  = "2023-06-01"
	// defaultMessagesPath is the Anthropic messages endpoint; a registry
	// entry may override it with a relative path or an absolute URL.
	defaultMessagesPath = "/v1/messages"
	// defaultMaxTokens mirrors the Python runtime: Anthropic requires
	// max_tokens, so a safe default is sent when the caller sets none.
	defaultMaxTokens = 4096
	// defaultTimeout mirrors the Python transport's 300s ceiling — a grounded
	// reasoning run can take well over a minute end to end.
	defaultTimeout = 300 * time.Second

	// Grounding server-tool defaults (Python GroundingSpec.effective_*).
	defaultGroundingToolType = "web_search_20250305"
	defaultGroundingToolName = "web_search"

	// providerErrorBodyLimit mirrors the Python runtime's 2048-char truncation
	// of the upstream error body.
	providerErrorBodyLimit = 2048
)

// ProviderError mirrors the Python runtime's ProviderError: a non-2xx
// response from the provider. Carries the provider's own HTTP status so the
// facade can relay it instead of flattening everything into one code.
type ProviderError struct {
	Status   int
	Endpoint string
	Body     string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("upstream %d from %s: %s", e.Status, e.Endpoint, e.Body)
}

// UpstreamStatus exposes the provider's HTTP status for status mapping.
func (e *ProviderError) UpstreamStatus() int { return e.Status }

// GroundingConfig configures the hosted web-search server tool
// (web_search_20250305) the adapter injects on a GROUNDED request. Nil means
// the registry entry configures no grounding (Python: spec.grounding is None).
type GroundingConfig struct {
	// ToolType overrides the server tool type (default web_search_20250305).
	ToolType string
	// ToolName overrides the tool name (default web_search).
	ToolName string
	// MaxUses bounds how many times the model may call the tool; 0 omits it.
	MaxUses int
	// ExtraFields merges verbatim into the tool object ("type" excluded).
	ExtraFields map[string]any
}

// tool renders the Anthropic server-tool object, mirroring the Python
// runtime's _grounding_tool for the messages surface.
func (g *GroundingConfig) tool() map[string]any {
	toolType := g.ToolType
	if toolType == "" {
		toolType = defaultGroundingToolType
	}
	toolName := g.ToolName
	if toolName == "" {
		toolName = defaultGroundingToolName
	}
	tool := map[string]any{"type": toolType, "name": toolName}
	if g.MaxUses > 0 {
		tool["max_uses"] = g.MaxUses
	}
	for key, value := range g.ExtraFields {
		if key == "type" {
			continue
		}
		tool[key] = value
	}
	return tool
}

// Client implements domain.VendorClient for Anthropic /v1/messages.
type Client struct {
	httpClient *http.Client
	secrets    domain.SecretClient
	endpoint   string
	version    string
	// messagesPath overrides the /v1/messages path: a relative path is
	// joined onto endpoint, an absolute http(s) URL is used verbatim
	// (Python _resolve_endpoint).
	messagesPath string
	// extraHeaders rides on every request (Python spec.extra_headers).
	extraHeaders map[string]string
	// grounding configures the web_search_20250305 server tool; nil disables.
	grounding *GroundingConfig
}

// Config groups construction inputs.
type Config struct {
	HTTPClient *http.Client
	Secrets    domain.SecretClient // required (BYOA-only)
	Endpoint   string              // optional override
	Version    string              // optional override — anthropic-version header
	// MessagesPath overrides the /v1/messages path (relative or absolute).
	MessagesPath string
	// ExtraHeaders merges into every request's headers.
	ExtraHeaders map[string]string
	// Grounding configures the hosted web-search server tool.
	Grounding *GroundingConfig
	// Timeout bounds each request when HTTPClient is not injected; default
	// 300s (the Python transport's ceiling).
	Timeout time.Duration
}

// New constructs a Client. Fail-loud on missing required fields.
func New(cfg Config) (*Client, error) {
	if cfg.Secrets == nil {
		return nil, fmt.Errorf("anthropic: SecretClient required")
	}
	hc := cfg.HTTPClient
	if hc == nil {
		timeout := cfg.Timeout
		if timeout <= 0 {
			timeout = defaultTimeout
		}
		hc = &http.Client{Timeout: timeout, Transport: newPooledTransport()}
	}
	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = defaultEndpoint
	}
	version := cfg.Version
	if version == "" {
		version = defaultVersion
	}
	return &Client{
		httpClient:   hc,
		secrets:      cfg.Secrets,
		endpoint:     endpoint,
		version:      version,
		messagesPath: cfg.MessagesPath,
		extraHeaders: cfg.ExtraHeaders,
		grounding:    cfg.Grounding,
	}, nil
}

// newPooledTransport builds the default connection-pooled transport with a
// layered timeout hierarchy (connection / response-header / overall). The
// Python reference pools one client per (base_url, auth, extra_headers) pair;
// the Go client is shared across tenants, so the pool is sized for
// concurrent use with a bounded idle lifetime.
func newPooledTransport() *http.Transport {
	return &http.Transport{
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ForceAttemptHTTP2:     true,
		DialContext: (&net.Dialer{
			Timeout: 10 * time.Second,
		}).DialContext,
	}
}

// Family — implements domain.VendorClient.
func (c *Client) Family() domain.VendorFamily { return domain.VendorFamilyAnthropic }

// Generate — implements domain.VendorClient.
func (c *Client) Generate(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	if err := domain.CheckContextCancellation(ctx); err != nil {
		return domain.VendorResponse{}, err
	}
	// BYOA-only — no global fallback.
	apiKey, err := c.secrets.ResolveByoaKey(ctx, req.TenantID, domain.VendorFamilyAnthropic)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: resolve BYOA: %w", err)
	}
	if apiKey == "" {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: tenant %q has no BYOA Anthropic key (no global fallback by policy)", req.TenantID)
	}

	url := resolveEndpoint(c.endpoint, c.messagesPath, defaultMessagesPath)
	body := buildAnthropicBody(req, c.grounding)
	rawBody, err := json.Marshal(body)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(rawBody))
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: build request: %w", err)
	}
	// Header precedence mirrors the Python runtime: extra headers first, then
	// the standard headers win (x-api-key + anthropic-version are set last so
	// a registry extra header cannot clobber the credential). This ordering is
	// a security boundary — see domain.GatewayControlledHeaders.
	httpReq.Header.Set("Content-Type", "application/json")
	for key, value := range c.extraHeaders {
		if domain.IsGatewayControlledHeader(key) {
			continue
		}
		httpReq.Header.Set(key, value)
	}
	httpReq.Header.Set("anthropic-version", c.version)
	// Anthropic API: x-api-key + anthropic-version (not Authorization Bearer).
	httpReq.Header.Set("x-api-key", apiKey)
	if req.Traceparent != "" {
		httpReq.Header.Set("traceparent", req.Traceparent)
	}
	if req.Tracestate != "" {
		httpReq.Header.Set("tracestate", req.Tracestate)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: HTTP do: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		return domain.VendorResponse{}, &ProviderError{
			Status:   httpResp.StatusCode,
			Endpoint: url,
			Body:     truncate(strings.TrimSpace(string(respBody)), providerErrorBodyLimit),
		}
	}
	var parsed anthropicResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: unmarshal response: %w", err)
	}
	return parsed.toDomain(string(req.LogicalModelID)), nil
}

// resolveEndpoint mirrors the Python runtime's _resolve_endpoint: an empty
// path falls back to defaultPath; an absolute http(s) URL is used verbatim;
// anything else is joined onto base.
func resolveEndpoint(base, path, defaultPath string) string {
	if path == "" {
		path = defaultPath
	}
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		return path
	}
	if base == "" {
		return path
	}
	return strings.TrimRight(base, "/") + "/" + strings.TrimLeft(path, "/")
}

// truncate mirrors the Python runtime's _truncate: text beyond limit is cut
// and suffixed with an ellipsis. Rune-based, like Python's len/slice, so a
// multi-byte character is never split.
func truncate(text string, limit int) string {
	runes := []rune(text)
	if len(runes) <= limit {
		return text
	}
	return string(runes[:limit]) + "…"
}

// ---------------------------------------------------------------------------
// Wire shapes — Anthropic /v1/messages.
// ---------------------------------------------------------------------------

type anthropicBody struct {
	Model     string          `json:"model"`
	MaxTokens int             `json:"max_tokens"`
	System    string          `json:"system,omitempty"`
	Messages  json.RawMessage `json:"messages"`
	// Tools carries the caller's tool declarations (genai shape, verbatim
	// passthrough like the Python runtime) with the grounding server tool
	// appended on a GROUNDED request.
	Tools json.RawMessage `json:"tools,omitempty"`
	// Generation-config breadth. temperature/top_p/stop_sequences are
	// Anthropic-native; n/seed/tool_choice/response_format pass through with
	// the Python runtime's coercion rules.
	Temperature    *float64        `json:"temperature,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	StopSequences  []string        `json:"stop_sequences,omitempty"`
	N              *int            `json:"n,omitempty"`
	Seed           *int            `json:"seed,omitempty"`
	ToolChoice     json.RawMessage `json:"tool_choice,omitempty"`
	ResponseFormat json.RawMessage `json:"response_format,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicResponse struct {
	Model      string                  `json:"model"`
	StopReason string                  `json:"stop_reason"`
	Content    []anthropicContentBlock `json:"content"`
	Usage      anthropicUsage          `json:"usage"`
}

// anthropicContentBlock covers the response block kinds the gateway reads:
// text (completion + citations) and server_tool_use (grounded search query).
type anthropicContentBlock struct {
	Type      string              `json:"type"`
	Text      string              `json:"text"`
	Citations []anthropicCitation `json:"citations"`
	Input     *anthropicToolInput `json:"input"`
}

type anthropicCitation struct {
	URL       string `json:"url"`
	Title     string `json:"title"`
	CitedText string `json:"cited_text"`
}

type anthropicToolInput struct {
	Query string `json:"query"`
}

type anthropicUsage struct {
	InputTokens              int64 `json:"input_tokens"`
	OutputTokens             int64 `json:"output_tokens"`
	CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
}

func buildAnthropicBody(req domain.VendorRequest, grounding *GroundingConfig) anthropicBody {
	body := anthropicBody{
		Model:     string(req.LogicalModelID),
		MaxTokens: defaultMaxTokens, // Anthropic API requires max_tokens; safe default.
		System:    req.SystemPrompt,
		Messages:  marshalMessages(req),
	}
	if strings.TrimSpace(req.ToolsJSON) != "" {
		body.Tools = json.RawMessage(req.ToolsJSON)
	}
	// Hosted web-search grounding: append the web_search_20250305 server tool
	// when the request's response modality is GROUNDED and the registry entry
	// configures grounding (Python: grounded and spec.grounding).
	if strings.EqualFold(req.ResponseModality, "GROUNDED") && grounding != nil {
		body.Tools = withGroundingTool(body.Tools, grounding)
	}
	applyGenerationConfig(&body, req.GenerationConfig)
	return body
}

// marshalMessages builds the messages array: the caller's ContentsJSON when
// it is a non-empty JSON array (verbatim passthrough, like the Python
// runtime), else the flat single-user-prompt fallback.
func marshalMessages(req domain.VendorRequest) json.RawMessage {
	if strings.TrimSpace(req.ContentsJSON) != "" {
		var msgs []json.RawMessage
		if err := json.Unmarshal([]byte(req.ContentsJSON), &msgs); err == nil && len(msgs) > 0 {
			return json.RawMessage(req.ContentsJSON)
		}
	}
	raw, err := json.Marshal([]anthropicMessage{{Role: "user", Content: req.Prompt}})
	if err != nil {
		return json.RawMessage(`[]`)
	}
	return raw
}

// withGroundingTool appends the grounding server tool to the caller's tools.
// A ToolsJSON that is not a JSON array is dropped (the provider would reject
// it anyway) so the grounding tool still reaches the model.
func withGroundingTool(tools json.RawMessage, grounding *GroundingConfig) json.RawMessage {
	toolJSON, err := json.Marshal(grounding.tool())
	if err != nil {
		return tools
	}
	arr := []json.RawMessage{}
	if len(tools) > 0 {
		if err := json.Unmarshal(tools, &arr); err != nil {
			arr = nil
		}
	}
	raw, err := json.Marshal(append(arr, toolJSON))
	if err != nil {
		return toolJSON
	}
	return raw
}

// applyGenerationConfig mirrors the Python runtime's _apply_generation_config
// for the messages surface: None values are skipped, numbers are coerced with
// _number semantics (bools rejected), stop/stop_sequences map onto Anthropic's
// stop_sequences list, and tool_choice/response_format pass through verbatim.
func applyGenerationConfig(body *anthropicBody, cfg map[string]any) {
	for key, value := range cfg {
		if value == nil {
			continue
		}
		switch key {
		case "temperature":
			if n, ok := number(value); ok {
				body.Temperature = &n
			}
		case "top_p":
			if n, ok := number(value); ok {
				body.TopP = &n
			}
		case "max_tokens", "max_completion_tokens":
			if n, ok := number(value); ok {
				body.MaxTokens = int(n)
			}
		case "n":
			if n, ok := number(value); ok {
				body.N = intPtr(int(n))
			}
		case "seed":
			if n, ok := number(value); ok {
				body.Seed = intPtr(int(n))
			}
		case "stop", "stop_sequences":
			body.StopSequences = appendStrings(body.StopSequences, value)
		case "tool_choice":
			if raw, err := json.Marshal(value); err == nil {
				body.ToolChoice = raw
			}
		case "response_format":
			if raw, err := json.Marshal(value); err == nil {
				body.ResponseFormat = raw
			}
		}
	}
}

// number coerces a GenerationConfig value to float64, mirroring the Python
// runtime's _number: bools are rejected, ints/floats accepted.
func number(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}

// appendStrings coerces a stop value to a string list, mirroring the Python
// runtime: a bare string becomes a one-element list; a list keeps only its
// string elements.
func appendStrings(dst []string, value any) []string {
	switch v := value.(type) {
	case string:
		return append(dst, v)
	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok {
				dst = append(dst, s)
			}
		}
	}
	return dst
}

func intPtr(n int) *int { return &n }

func (r anthropicResponse) toDomain(fallbackModel string) domain.VendorResponse {
	// Cache accounting, mirroring the Python runtime's _messages_to_result:
	// cache writes are excluded from the plain-input total and billed at the
	// cache-write rate; cache reads are billed at the discounted cache rate.
	plainInput := r.Usage.InputTokens - r.Usage.CacheCreationInputTokens
	if plainInput < 0 {
		plainInput = 0
	}
	resp := domain.VendorResponse{
		Usage: domain.TokenUsage{
			InputTokens:  r.Usage.InputTokens,
			OutputTokens: r.Usage.OutputTokens,
			CachedTokens: r.Usage.CacheReadInputTokens,
			CostMicros:   anthropicCostMicros(fallbackModel, plainInput, r.Usage.OutputTokens, r.Usage.CacheReadInputTokens, r.Usage.CacheCreationInputTokens),
		},
		ModelVersion: r.Model,
	}
	if resp.ModelVersion == "" {
		resp.ModelVersion = fallbackModel
	}
	for _, block := range r.Content {
		switch block.Type {
		case "text":
			resp.Completion += block.Text
			for _, citation := range block.Citations {
				if citation.URL == "" {
					continue
				}
				resp.Citations = append(resp.Citations, domain.Citation{
					URL:     citation.URL,
					Title:   citation.Title,
					Snippet: citation.CitedText,
				})
			}
		case "server_tool_use":
			if block.Input != nil && block.Input.Query != "" {
				resp.SearchQueries = append(resp.SearchQueries, block.Input.Query)
			}
		}
	}
	resp.FinishReason = mapAnthropicStopReason(r.StopReason)
	return resp
}

func mapAnthropicStopReason(s string) domain.FinishReason {
	switch s {
	case "end_turn", "stop_sequence":
		return domain.FinishReasonComplete
	case "max_tokens":
		return domain.FinishReasonMaxTokens
	case "refusal":
		return domain.FinishReasonModelArmorBlock
	default:
		return domain.FinishReasonUnspecified
	}
}

// anthropicPrice is one model's rates in micro-USD per 1M tokens. Cache reads
// bill at the discounted cache rate and cache writes at 1.25x input —
// Anthropic's published prompt-caching rates. Hard-coded for Phase 2.2;
// Phase 4 follow-up moves this into GCS-managed YAML config.
type anthropicPrice struct {
	InputMicrosPer1M      int64
	OutputMicrosPer1M     int64
	CachedMicrosPer1M     int64
	CacheWriteMicrosPer1M int64
}

var anthropicPricing = map[string]anthropicPrice{
	"claude-opus-4-7":   {15_000_000, 75_000_000, 1_500_000, 18_750_000},
	"claude-opus-4-6":   {15_000_000, 75_000_000, 1_500_000, 18_750_000},
	"claude-sonnet-4-6": {3_000_000, 15_000_000, 300_000, 3_750_000},
	"claude-haiku-4-5":  {1_000_000, 5_000_000, 100_000, 1_250_000},
}

// anthropicCostMicros mirrors the Python ModelSpec.cost_micros: term-wise
// floored micro-USD — billable input (plain input minus cache reads) at the
// input rate, cache reads at the cache rate, cache writes at the cache-write
// rate, output at the output rate.
func anthropicCostMicros(model string, plainInput, outputTokens, cachedTokens, cacheWriteTokens int64) int64 {
	p, ok := anthropicPricing[model]
	if !ok {
		return 0
	}
	billable := plainInput - cachedTokens
	if billable < 0 {
		billable = 0
	}
	in := (billable * p.InputMicrosPer1M) / 1_000_000
	cached := (cachedTokens * p.CachedMicrosPer1M) / 1_000_000
	writes := (cacheWriteTokens * p.CacheWriteMicrosPer1M) / 1_000_000
	out := (outputTokens * p.OutputMicrosPer1M) / 1_000_000
	return in + cached + writes + out
}
