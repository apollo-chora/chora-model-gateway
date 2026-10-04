// Package openai implements the vendor ports against the OpenAI HTTP API
// shape.
//
// One adapter instance serves every registry entry whose provider is
// "openai": the upstream model name, base URL, endpoint paths and
// credential all arrive per-request on domain.VendorRequest. That is what
// lets a user-supplied registry point different entries at OpenAI, a
// colleague's vLLM, Ollama, LM Studio or llama.cpp's server without any
// code change — they all speak this spec.
//
// Three surfaces are implemented, all under the same provider:
//
//	POST {chat_completions_path}  → Generate   (text + tools + vision)
//	POST {images_path}            → Generate   (response_modality IMAGE)
//	POST {embeddings_path}        → EmbedText
package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/apollo-chora/chora-model-gateway/internal/config"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// Client implements domain.VendorClient and domain.EmbeddingClient for the
// OpenAI-compatible wire shape.
type Client struct {
	httpClient *http.Client
	pricing    *config.Registry
}

// Config groups construction inputs.
type Config struct {
	// HTTPClient is optional; defaults to a client with no timeout of its own
	// so the caller's context governs the deadline.
	HTTPClient *http.Client

	// Pricing is the registry consulted for the per-model cost table. Optional
	// — a nil registry means every call is recorded at zero cost, which is
	// the honest answer for a model the operator has not priced.
	Pricing *config.Registry
}

// New constructs a Client.
func New(cfg Config) (*Client, error) {
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{}
	}
	return &Client{httpClient: hc, pricing: cfg.Pricing}, nil
}

// Family — implements domain.VendorClient.
func (c *Client) Family() domain.VendorFamily { return domain.VendorFamilyOpenAI }

// Generate — implements domain.VendorClient. Routes on the requested output
// modality: IMAGE to the images surface, GROUNDED to whichever search surface
// the registry entry configured, everything else to chat completions.
func (c *Client) Generate(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	switch {
	case strings.EqualFold(req.ResponseModality, domain.ModalityImage):
		return c.generateImage(ctx, req)
	case strings.EqualFold(req.ResponseModality, domain.ModalityGrounded):
		// Route on the surface the REGISTRY configured, not on a hardcoded
		// preference for /responses. An entry that grounds on
		// chat_completions must go there, or the search tool lands on an
		// endpoint the provider does not read it from.
		if req.Target.Grounding != nil &&
			req.Target.Grounding.EffectiveSurface(req.Target.Vendor) == domain.SurfaceChatCompletions {
			return c.generateChat(ctx, req)
		}
		return c.generateGrounded(ctx, req)
	default:
		return c.generateChat(ctx, req)
	}
}

func (c *Client) generateChat(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	body := buildChatBody(req)

	raw, err := json.Marshal(body)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: marshal request: %w", err)
	}

	var parsed chatResponse
	if err := c.do(ctx, req, req.Target.ChatCompletionsURL(), raw, &parsed); err != nil {
		return domain.VendorResponse{}, err
	}
	return c.chatResponseToDomain(req, parsed), nil
}

func (c *Client) generateImage(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	prompt := req.Prompt
	if prompt == "" {
		return domain.VendorResponse{}, fmt.Errorf("openai: image generation requires a prompt")
	}

	body := map[string]any{
		"model":  req.Target.UpstreamModel,
		"prompt": prompt,
		"n":      1,
	}
	// Only forward generation knobs the images surface actually accepts;
	// sending `temperature` to an images endpoint is a 400 on OpenAI.
	for _, key := range []string{"size", "quality", "style", "n", "response_format", "user"} {
		if v, ok := req.GenerationConfig[key]; ok && v != nil {
			body[key] = v
		}
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: marshal image request: %w", err)
	}

	var parsed imagesResponse
	if err := c.do(ctx, req, req.Target.ImagesURL(), raw, &parsed); err != nil {
		return domain.VendorResponse{}, err
	}
	return c.imageResponseToDomain(ctx, req, parsed)
}

// EmbedText — implements domain.EmbeddingClient.
func (c *Client) EmbedText(ctx context.Context, req domain.EmbedVendorRequest) (domain.EmbedVendorResponse, error) {
	model := req.Target.UpstreamModel
	if model == "" {
		model = string(req.LogicalModelID)
	}
	body := map[string]any{"model": model, "input": req.Text}
	if req.OutputDimensions > 0 {
		body["dimensions"] = req.OutputDimensions
	}

	raw, err := json.Marshal(body)
	if err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("openai: marshal embed request: %w", err)
	}

	// Embed requests ride a VendorRequest-shaped envelope purely to reuse
	// the shared HTTP path; the target and credential are what matter.
	var parsed embeddingsResponse
	if err := c.do(ctx, domain.VendorRequest{
		Target:      req.Target,
		Credential:  req.Credential,
		Traceparent: req.Traceparent,
	}, req.Target.EmbeddingsURL(), raw, &parsed); err != nil {
		return domain.EmbedVendorResponse{}, err
	}
	if len(parsed.Data) == 0 {
		return domain.EmbedVendorResponse{}, fmt.Errorf("openai: embeddings response carried no data")
	}
	return domain.EmbedVendorResponse{
		Values:       parsed.Data[0].Embedding,
		ModelVersion: firstNonEmpty(parsed.Model, model),
		InputTokens:  parsed.Usage.PromptTokens,
	}, nil
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

// do performs one upstream call: builds the request, attaches auth +
// tracing, decodes the response, and normalises the error shape.
func (c *Client) do(ctx context.Context, req domain.VendorRequest, url string, rawBody []byte, out any) error {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(rawBody))
	if err != nil {
		return fmt.Errorf("openai: build request for %s: %w", url, err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if req.Credential != "" {
		httpReq.Header.Set("Authorization", "Bearer "+req.Credential)
	}
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
		return fmt.Errorf("openai: POST %s: %w", url, err)
	}
	defer func() { _ = httpResp.Body.Close() }()

	respBody, readErr := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		return &UpstreamError{
			StatusCode: httpResp.StatusCode,
			Endpoint:   url,
			Body:       truncate(strings.TrimSpace(string(respBody)), 2048),
		}
	}
	if readErr != nil {
		return fmt.Errorf("openai: read response from %s: %w", url, readErr)
	}
	if err := json.Unmarshal(respBody, out); err != nil {
		return fmt.Errorf("openai: decode response from %s: %w (body: %s)", url, err, truncate(strings.TrimSpace(string(respBody)), 512))
	}
	return nil
}

// UpstreamError is a non-2xx response from the provider. It is exported so
// the HTTP facade can decide a status code: a 401 or 429 from upstream is a
// gateway misconfiguration or a rate limit, and passing the provider's own
// status through is more useful than flattening everything to 502.
type UpstreamError struct {
	StatusCode int
	Endpoint   string
	Body       string
}

// UpstreamStatusCode exposes the provider's own HTTP status so the HTTP
// facade can relay it instead of flattening every provider failure to 502.
func (e *UpstreamError) UpstreamStatusCode() int { return e.StatusCode }

func (e *UpstreamError) Error() string {
	return fmt.Sprintf("openai: upstream %d from %s: %s", e.StatusCode, e.Endpoint, e.Body)
}

// ---------------------------------------------------------------------------
// Chat completions wire shapes
// ---------------------------------------------------------------------------

type chatMessage struct {
	Role       string     `json:"role"`
	Content    any        `json:"content,omitempty"`
	Name       string     `json:"name,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
	ToolCalls  []toolCall `json:"tool_calls,omitempty"`
	// Annotations carry url_citation entries when a search tool ran on the
	// chat-completions surface.
	Annotations []responsesAnnotation `json:"annotations,omitempty"`
}

type toolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type chatBody struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream,omitempty"`

	// Generation knobs are forwarded verbatim from the caller's config so a
	// self-hosted server's extensions (repetition_penalty, top_k,
	// presence_penalty, ...) pass through untouched.
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	N           *int     `json:"n,omitempty"`
	Seed        *int     `json:"seed,omitempty"`
	Tools       []any    `json:"tools,omitempty"`
	ToolChoice  any      `json:"tool_choice,omitempty"`
	ResponseFmt any      `json:"response_format,omitempty"`
}

type chatResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message      chatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
		PromptDetails    struct {
			CachedTokens int64 `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionDetails struct {
			ReasoningTokens int64 `json:"reasoning_tokens"`
		} `json:"completion_tokens_details"`
	} `json:"usage"`
}

func buildChatBody(req domain.VendorRequest) chatBody {
	body := chatBody{
		Model: req.Target.UpstreamModel,
		// Streaming is not implemented on the gateway; refusing the knob is
		// better than sending stream:true and buffering the whole SSE stream
		// as a string.
		Stream: false,
	}

	// A tool-calling turn carries a full conversation, so it wins over the
	// single-prompt path.
	if req.ContentsJSON != "" {
		var msgs []chatMessage
		if err := json.Unmarshal([]byte(req.ContentsJSON), &msgs); err != nil {
			// A malformed conversation is a caller bug; fall back to the bare
			// prompt rather than dropping the call.
			body.Messages = []chatMessage{{Role: "user", Content: req.Prompt}}
		} else {
			body.Messages = msgs
		}
	} else {
		if req.SystemPrompt != "" {
			body.Messages = append(body.Messages, chatMessage{Role: "system", Content: req.SystemPrompt})
		}
		body.Messages = append(body.Messages, chatMessage{Role: "user", Content: req.Prompt})
	}

	if req.ToolsJSON != "" {
		// Passed through as raw JSON: the tools schema is provider-defined
		// and unmodelling it here would only lose fields.
		var tools []json.RawMessage
		if err := json.Unmarshal([]byte(req.ToolsJSON), &tools); err == nil {
			body.Tools = make([]any, 0, len(tools))
			for _, t := range tools {
				var v any
				if err := json.Unmarshal(t, &v); err == nil {
					body.Tools = append(body.Tools, v)
				}
			}
		}
	}

	// A caller asking for grounding on the chat-completions surface gets the
	// registry's search tool attached here rather than being told to move to
	// /v1/responses. Only honoured when the entry configured the
	// chat_completions grounding surface.
	if strings.EqualFold(req.ResponseModality, domain.ModalityGrounded) && req.Target.Grounding != nil {
		if req.Target.Grounding.EffectiveSurface(req.Target.Vendor) == domain.SurfaceChatCompletions {
			body.Tools = append(body.Tools, buildGroundingTools(req)...)
		}
	}

	applyGenerationConfig(&body, req.GenerationConfig)
	return body
}

func applyGenerationConfig(body *chatBody, cfg map[string]any) {
	for k, v := range cfg {
		switch k {
		case "temperature":
			if f, ok := numberFrom(v); ok {
				t := f
				body.Temperature = &t
			}
		case "top_p":
			if f, ok := numberFrom(v); ok {
				p := f
				body.TopP = &p
			}
		case "max_tokens", "max_completion_tokens":
			if f, ok := numberFrom(v); ok {
				n := int(f)
				body.MaxTokens = &n
			}
		case "n":
			if f, ok := numberFrom(v); ok {
				n := int(f)
				body.N = &n
			}
		case "seed":
			if f, ok := numberFrom(v); ok {
				n := int(f)
				body.Seed = &n
			}
		case "stop":
			switch s := v.(type) {
			case string:
				body.Stop = []string{s}
			case []string:
				body.Stop = s
			case []any:
				for _, item := range s {
					if str, ok := item.(string); ok {
						body.Stop = append(body.Stop, str)
					}
				}
			}
		case "tool_choice":
			body.ToolChoice = v
		case "response_format":
			body.ResponseFmt = v
		case "stream":
			// Explicitly refused above; recorded here so the omission is
			// traceable rather than looking like an oversight.
		}
	}
}

func (c *Client) chatResponseToDomain(req domain.VendorRequest, r chatResponse) domain.VendorResponse {
	model := firstNonEmpty(r.Model, req.Target.UpstreamModel)
	usage := domain.TokenUsage{
		InputTokens:  r.Usage.PromptTokens,
		OutputTokens: r.Usage.CompletionTokens,
		CachedTokens: r.Usage.PromptDetails.CachedTokens,
		CostMicros:   c.costMicros(req.Target.LogicalModelID, r.Usage.PromptTokens, r.Usage.CompletionTokens, r.Usage.PromptDetails.CachedTokens),
	}

	out := domain.VendorResponse{
		Usage:        usage,
		ModelVersion: model,
		FinishReason: domain.FinishReasonComplete,
	}
	if len(r.Choices) == 0 {
		out.FinishReason = domain.FinishReasonUnspecified
		out.FinishDetail = "upstream returned no choices"
		return out
	}
	choice := r.Choices[0]
	out.Completion = contentToString(choice.Message.Content)
	out.FinishReason = mapFinishReason(choice.FinishReason)
	out.Citations = citationsFromAnnotations(choice.Message.Annotations)

	if len(choice.Message.ToolCalls) > 0 {
		calls := make([]map[string]any, 0, len(choice.Message.ToolCalls))
		for _, tc := range choice.Message.ToolCalls {
			calls = append(calls, map[string]any{
				"id":   tc.ID,
				"type": "function",
				"function": map[string]any{
					"name":      tc.Function.Name,
					"arguments": tc.Function.Arguments,
				},
			})
		}
		if encoded, err := json.Marshal(calls); err == nil {
			out.ToolCallsJSON = string(encoded)
		}
	}
	return out
}

func mapFinishReason(s string) domain.FinishReason {
	switch s {
	case "stop", "tool_calls", "function_call":
		return domain.FinishReasonComplete
	case "length":
		return domain.FinishReasonMaxTokens
	case "content_filter":
		return domain.FinishReasonUnspecified
	default:
		return domain.FinishReasonComplete
	}
}

// ---------------------------------------------------------------------------
// Images wire shapes
// ---------------------------------------------------------------------------

type imagesResponse struct {
	Created int64 `json:"created"`
	Data    []struct {
		B64JSON       string `json:"b64_json"`
		URL           string `json:"url"`
		RevisedPrompt string `json:"revised_prompt"`
	} `json:"data"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

func (c *Client) imageResponseToDomain(ctx context.Context, req domain.VendorRequest, r imagesResponse) (domain.VendorResponse, error) {
	out := domain.VendorResponse{
		ModelVersion: req.Target.UpstreamModel,
		FinishReason: domain.FinishReasonComplete,
		Usage: domain.TokenUsage{
			InputTokens:  r.Usage.PromptTokens,
			OutputTokens: r.Usage.CompletionTokens,
			// Images carry no token price in any provider's response; the
			// cost is the operator's configured per-call figure, which the
			// registry models as zero unless the operator sets it.
			CostMicros: 0,
		},
	}
	if len(r.Data) == 0 {
		out.FinishReason = domain.FinishReasonUnspecified
		out.FinishDetail = "upstream returned no image data"
		return out, nil
	}

	first := r.Data[0]
	out.ImageRevisedPrompt = first.RevisedPrompt

	switch {
	case first.B64JSON != "":
		decoded, err := base64.StdEncoding.DecodeString(first.B64JSON)
		if err != nil {
			// A malformed blob is a provider-contract violation, not an empty
			// image; fail loud rather than returning a zero-byte picture.
			return domain.VendorResponse{}, fmt.Errorf("openai: decode image b64_json: %w", err)
		}
		out.ImageBytes = decoded
		// Sniffed, not assumed. The provider sends no Content-Type for an
		// inline image, and not every provider returns PNG despite what the
		// OpenAI docs say — api.meta.ai's muse-image-1.0 returns WebP.
		out.ImageMIMEType = sniffImageMIME(decoded)
	case first.URL != "":
		// The provider returned a URL instead of bytes. Fetching it here
		// keeps the facade's contract simple: the caller always gets bytes,
		// matching what the gRPC surface has always returned.
		imgBytes, mime, err := c.fetchImage(ctx, first.URL, req)
		if err != nil {
			return domain.VendorResponse{}, err
		}
		out.ImageBytes = imgBytes
		out.ImageMIMEType = mime
	default:
		return domain.VendorResponse{}, fmt.Errorf("openai: image response carried neither b64_json nor url")
	}
	return out, nil
}

func (c *Client) fetchImage(ctx context.Context, url string, req domain.VendorRequest) ([]byte, string, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, "", fmt.Errorf("openai: build image fetch request: %w", err)
	}
	// The download URL is provider-minted and pre-signed, so no credential is
	// attached — sending one can leak it to a third-party host.
	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, "", fmt.Errorf("openai: fetch image: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	if httpResp.StatusCode >= 400 {
		return nil, "", fmt.Errorf("openai: fetch image: upstream %d", httpResp.StatusCode)
	}
	// Bound the read: an unbounded fetch of an untrusted URL is a memory
	// exhaustion vector.
	body, err := io.ReadAll(io.LimitReader(httpResp.Body, 32<<20))
	if err != nil {
		return nil, "", fmt.Errorf("openai: read image: %w", err)
	}
	mime := httpResp.Header.Get("Content-Type")
	if mime == "" {
		mime = "image/png"
	}
	return body, mime, nil
}

// ---------------------------------------------------------------------------
// Embeddings wire shapes
// ---------------------------------------------------------------------------

type embeddingsResponse struct {
	Model string `json:"model"`
	Data  []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Usage struct {
		PromptTokens int64 `json:"prompt_tokens"`
		TotalTokens  int64 `json:"total_tokens"`
	} `json:"usage"`
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// costMicros prices a call from the registry. An unpriced model costs zero,
// which is the honest answer for a self-hosted or operator-supplied model.
func (c *Client) costMicros(id domain.LogicalModelID, input, output, cached int64) int64 {
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
	micros += (output * p.OutputPerMTokUSDMicros) / 1_000_000
	return micros
}

// contentToString flattens the OpenAI content union, which is a plain string
// on most servers and an array of typed parts on the vision-capable ones.
func contentToString(content any) string {
	switch v := content.(type) {
	case nil:
		return ""
	case string:
		return v
	case []any:
		var b strings.Builder
		for _, part := range v {
			m, ok := part.(map[string]any)
			if !ok {
				continue
			}
			if text, ok := m["text"].(string); ok {
				b.WriteString(text)
			}
		}
		return b.String()
	default:
		return ""
	}
}

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

// Compile-time proof the adapter satisfies both ports.
var (
	_ domain.VendorClient    = (*Client)(nil)
	_ domain.EmbeddingClient = (*Client)(nil)
)
