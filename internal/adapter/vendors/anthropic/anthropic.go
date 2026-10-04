// Package anthropic implements domain.VendorClient against the Anthropic
// /v1/messages API.
//
// Like the OpenAI adapter, one Client instance serves every registry entry
// whose provider is "anthropic": the upstream model name, base URL and
// credential arrive per-request. Unlike OpenAI, the messages API has no
// image-generation surface, so a registry entry pointing at Anthropic must
// not advertise the "image" capability.
package anthropic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/apollo-chora/chora-model-gateway/internal/config"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// defaultAPIVersion is the Anthropic API version header value. Pinned here
// because it is a protocol constant, not a deployment setting.
const defaultAPIVersion = "2023-06-01"

// Client implements domain.VendorClient for Anthropic /v1/messages.
type Client struct {
	httpClient *http.Client
	pricing    *config.Registry
	apiVersion string
}

// Config groups construction inputs.
type Config struct {
	// HTTPClient is optional; the caller's context governs the deadline.
	HTTPClient *http.Client

	// Pricing is the registry consulted for the per-model cost table.
	// Optional — a nil registry records every call at zero cost.
	Pricing *config.Registry

	// APIVersion overrides the anthropic-version header. Optional.
	APIVersion string
}

// New constructs a Client.
func New(cfg Config) (*Client, error) {
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{}
	}
	version := cfg.APIVersion
	if version == "" {
		version = defaultAPIVersion
	}
	return &Client{httpClient: hc, pricing: cfg.Pricing, apiVersion: version}, nil
}

// Family — implements domain.VendorClient.
func (c *Client) Family() domain.VendorFamily { return domain.VendorFamilyAnthropic }

// Generate — implements domain.VendorClient.
func (c *Client) Generate(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	if strings.EqualFold(req.ResponseModality, domain.ModalityImage) {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: the messages API has no image-generation surface; model %q cannot serve response_modality=IMAGE", req.Target.LogicalModelID)
	}

	rawBody, err := json.Marshal(buildBody(req))
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, messagesURL(req.Target), bytes.NewReader(rawBody))
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: build request: %w", err)
	}
	// Anthropic authenticates with x-api-key, not an Authorization bearer.
	// Sent only when a credential was resolved: a credential-less target must
	// not ship `x-api-key: ""`, which a strict server treats differently from
	// an absent header.
	if req.Credential != "" {
		httpReq.Header.Set("x-api-key", req.Credential)
	}
	httpReq.Header.Set("anthropic-version", c.apiVersion)
	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range req.ExtraHeaders {
		httpReq.Header.Set(k, v)
	}
	if req.Traceparent != "" {
		httpReq.Header.Set("traceparent", req.Traceparent)
	}
	if req.Tracestate != "" {
		httpReq.Header.Set("tracestate", req.Tracestate)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: POST %s: %w", messagesURL(req.Target), err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	respBody, readErr := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		return domain.VendorResponse{}, &UpstreamError{
			StatusCode: httpResp.StatusCode,
			Endpoint:   messagesURL(req.Target),
			Body:       truncate(strings.TrimSpace(string(respBody)), 2048),
		}
	}
	if readErr != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: read response: %w", readErr)
	}

	var parsed messagesResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return domain.VendorResponse{}, fmt.Errorf("anthropic: decode response: %w (body: %s)", err, truncate(strings.TrimSpace(string(respBody)), 512))
	}
	return c.toDomain(req, parsed), nil
}

// messagesURL is the Anthropic messages endpoint for a resolved target. The
// Anthropic API has no images or embeddings surface, so the messages path is
// the only one consulted. It is registry-configurable for the benefit of a
// gateway that proxies Anthropic-shaped traffic at a non-standard path.
func messagesURL(t domain.TargetModel) string {
	return t.MessagesURL()
}

// ---------------------------------------------------------------------------
// Wire shapes
// ---------------------------------------------------------------------------

type body struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content any    `json:"content"`
	} `json:"messages"`
	System      string   `json:"system,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	StopSeqs    []string `json:"stop_sequences,omitempty"`
	Stream      bool     `json:"stream,omitempty"`
	Tools       []any    `json:"tools,omitempty"`
}

// contentBlock is one block of the reply. Anthropic's content array is
// heterogeneous: text blocks carry the answer and its citations, while
// server_tool_use / web_search_tool_result blocks record the search itself.
type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text"`

	// Citations are attached to the text block they support. Only present when
	// a server-side search or fetch ran.
	Citations []messageCitation `json:"citations"`

	// Input is populated on a server_tool_use block and carries what the model
	// asked for — for a web search that is the query string.
	Input struct {
		Query string `json:"query"`
	} `json:"input"`
}

// messageCitation is one source Anthropic consulted.
type messageCitation struct {
	Type      string `json:"type"`
	URL       string `json:"url"`
	Title     string `json:"title"`
	CitedText string `json:"cited_text"`
}

// messagesResponse is the subset of the Messages payload the gateway reads.
// The inner types are NAMED rather than inline anonymous structs so a test can
// build a reply without restating the whole nested shape.
type messagesResponse struct {
	Model      string         `json:"model"`
	StopReason string         `json:"stop_reason"`
	Content    []contentBlock `json:"content"`
	Usage      struct {
		InputTokens          int64 `json:"input_tokens"`
		OutputTokens         int64 `json:"output_tokens"`
		CacheReadInputTokens int64 `json:"cache_read_input_tokens"`
		// Cache-write tokens are billed at a premium over plain input, so
		// dropping them would systematically under-bill a prompt-caching
		// workload. They are carried on the request so the cost calculator
		// can price them at the cache-write rate.
		CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

func buildBody(req domain.VendorRequest) body {
	out := body{
		Model:  req.Target.UpstreamModel,
		System: req.SystemPrompt,
		// max_tokens is REQUIRED by the Anthropic API, unlike OpenAI's where
		// it is optional. Fall back to the registry's ceiling, then to a
		// conservative default so a request never goes out malformed.
		MaxTokens: intPtr(req.Target.MaxOutputTokens),
	}

	if req.ContentsJSON != "" {
		var msgs []struct {
			Role    string `json:"role"`
			Content any    `json:"content"`
		}
		if err := json.Unmarshal([]byte(req.ContentsJSON), &msgs); err == nil && len(msgs) > 0 {
			out.Messages = msgs
		}
	}
	if len(out.Messages) == 0 {
		out.Messages = []struct {
			Role    string `json:"role"`
			Content any    `json:"content"`
		}{{Role: "user", Content: req.Prompt}}
	}

	if req.ToolsJSON != "" {
		var tools []json.RawMessage
		if err := json.Unmarshal([]byte(req.ToolsJSON), &tools); err == nil {
			out.Tools = make([]any, 0, len(tools))
			for _, t := range tools {
				var v any
				if err := json.Unmarshal(t, &v); err == nil {
					out.Tools = append(out.Tools, v)
				}
			}
		}
	}

	// Hosted web search is a SERVER tool on this same endpoint, so a grounded
	// call needs no separate request path — only the tools array changes. The
	// tool shape is Anthropic's, not OpenAI's, which is exactly why the
	// gateway normalises rather than pass-through: the caller asks for
	// "web_search" and the registry decides what actually goes on the wire.
	if strings.EqualFold(req.ResponseModality, domain.ModalityGrounded) && req.Target.Grounding != nil {
		out.Tools = append(out.Tools, buildWebSearchTool(req.Target.Grounding))
	}

	for k, v := range req.GenerationConfig {
		switch k {
		case "temperature":
			if f, ok := numberFrom(v); ok {
				t := f
				out.Temperature = &t
			}
		case "top_p":
			if f, ok := numberFrom(v); ok {
				p := f
				out.TopP = &p
			}
		case "max_tokens", "max_completion_tokens":
			if f, ok := numberFrom(v); ok {
				out.MaxTokens = intPtr(int(f))
			}
		case "stop", "stop_sequences":
			switch s := v.(type) {
			case string:
				out.StopSeqs = []string{s}
			case []string:
				out.StopSeqs = s
			case []any:
				for _, item := range s {
					if str, ok := item.(string); ok {
						out.StopSeqs = append(out.StopSeqs, str)
					}
				}
			}
		}
	}
	return out
}

func (c *Client) toDomain(req domain.VendorRequest, r messagesResponse) domain.VendorResponse {
	cached := r.Usage.CacheReadInputTokens
	// Anthropic reports cache WRITES as a subset of input_tokens, so the
	// plain-input rate must not be applied to them twice.
	cacheWrites := r.Usage.CacheCreationInputTokens
	plainInput := r.Usage.InputTokens - cacheWrites
	if plainInput < 0 {
		plainInput = 0
	}
	out := domain.VendorResponse{
		ModelVersion: firstNonEmpty(r.Model, req.Target.UpstreamModel),
		Usage: domain.TokenUsage{
			InputTokens:  r.Usage.InputTokens,
			OutputTokens: r.Usage.OutputTokens,
			CachedTokens: cached,
			CostMicros: c.costMicros(req.Target.LogicalModelID,
				plainInput, r.Usage.OutputTokens, cached, cacheWrites),
		},
		FinishReason: mapStopReason(r.StopReason),
	}
	for _, block := range r.Content {
		switch block.Type {
		case "text":
			out.Completion += block.Text
			// Anthropic carries character offsets nowhere, so only the URL,
			// title and cited passage survive the normalisation.
			for _, cite := range block.Citations {
				if cite.URL == "" {
					continue
				}
				out.Citations = append(out.Citations, domain.GroundingCitation{
					URL:     cite.URL,
					Title:   cite.Title,
					Snippet: cite.CitedText,
				})
			}
		case "server_tool_use":
			// The block records what the model asked for. Surfacing it as a
			// query is a reasonable approximation and is better than dropping
			// the only evidence the search actually ran.
			if block.Input.Query != "" {
				out.SearchQueries = append(out.SearchQueries, block.Input.Query)
			}
		}
	}
	return out
}

// buildWebSearchTool renders Anthropic's server tool.
//
// Anthropic requires BOTH a `type` discriminator AND a `name`, and `max_uses`
// bounds the number of searches — which matters because web search is priced
// per search, not per token, so an unbounded tool is unbounded spend.
func buildWebSearchTool(g *domain.Grounding) map[string]any {
	tool := map[string]any{
		"type": g.EffectiveToolType(domain.SurfaceMessages),
		"name": g.EffectiveToolName(),
	}
	if g.MaxUses > 0 {
		tool["max_uses"] = g.MaxUses
	}
	for k, v := range g.ExtraToolFields {
		// type and name are load-bearing for Anthropic's server-tool
		// dispatch, so a stray registry key cannot override them.
		if k == "type" || k == "name" {
			continue
		}
		tool[k] = v
	}
	return tool
}

func mapStopReason(s string) domain.FinishReason {
	switch s {
	case "end_turn", "stop_sequence", "tool_use":
		return domain.FinishReasonComplete
	case "max_tokens":
		return domain.FinishReasonMaxTokens
	default:
		return domain.FinishReasonComplete
	}
}

// costMicros prices a call from the registry. input excludes both cached-read
// and cache-write tokens; the caller splits them out because the two carry
// different rates. An unpriced model costs zero, which is the honest answer
// for a self-hosted or operator-supplied model.
func (c *Client) costMicros(id domain.LogicalModelID, input, output, cached, cacheWrites int64) int64 {
	if c.pricing == nil {
		return 0
	}
	p, ok := c.pricing.PricingFor(id)
	if !ok {
		return 0
	}
	billableInput := input - cached
	if billableInput < 0 {
		billableInput = 0
	}
	micros := (billableInput * p.InputPerMTokUSDMicros) / 1_000_000
	micros += (cached * p.CachedPerMTokUSDMicros) / 1_000_000
	micros += (cacheWrites * p.CacheWritePerMTokUSDMicros) / 1_000_000
	micros += (output * p.OutputPerMTokUSDMicros) / 1_000_000
	return micros
}

// UpstreamError is a non-2xx response from the provider.
type UpstreamError struct {
	StatusCode int
	Endpoint   string
	Body       string
}

// UpstreamStatusCode exposes the provider's own HTTP status so the HTTP
// facade can relay it instead of flattening every provider failure to 502.
func (e *UpstreamError) UpstreamStatusCode() int { return e.StatusCode }

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("anthropic: upstream %d from %s: %s", e.StatusCode, e.Endpoint, e.Body)
}

// req.MaxOutputTokens is a convenience accessor: the registry ceiling, or a
// conservative default when the entry leaves it open.
const defaultMaxTokens = 4096

func numberFrom(v any) (float64, bool) {
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

func intPtr(n int) *int {
	if n <= 0 {
		n = defaultMaxTokens
	}
	return &n
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

var _ domain.VendorClient = (*Client)(nil)
