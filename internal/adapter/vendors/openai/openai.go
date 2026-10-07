// Package openai implements domain.VendorClient against any OpenAI-
// compatible API surface. Used for:
//   - external OpenAI BYOA (api.openai.com/v1)
//   - self-hosted Gemma via vLLM (Phase 3) — vLLM exposes the OpenAI
//     chat-completions spec verbatim, so the same adapter dispatches
//     to either without code changes.
//   - many other vendors (Groq, Together, Fireworks, ...) that adopt
//     the OpenAI shape.
//
// Per ADR-163 the adapter resolves API keys via the SecretClient port:
//   tenant BYOA key first → global fallback (allow-list) second.
//
// The adapter supports three OpenAI surfaces:
//   - chat completions (POST /v1/chat/completions) — text + tool calls
//   - responses (POST /v1/responses) — hosted web search + structured output
//   - images (POST /v1/images/generations) — image generation
//   - embeddings (POST /v1/embeddings) — text embeddings
package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

const defaultEndpoint = "https://api.openai.com"

// maxImageFetchBytes is the 32 MiB cap on a fetched image URL response.
const maxImageFetchBytes = 32 << 20

// ProviderError is a non-2xx response from the provider. Carries the
// provider's own HTTP status + endpoint + body so the service layer can
// relay it instead of flattening everything.
type ProviderError struct {
	Status   int
	Endpoint string
	Body     string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("upstream %d from %s: %s", e.Status, e.Endpoint, e.Body)
}

// UpstreamStatus returns the provider's HTTP status code.
func (e *ProviderError) UpstreamStatus() int { return e.Status }

// Client implements domain.VendorClient for OpenAI-compatible APIs.
type Client struct {
	httpClient *http.Client
	secrets    domain.SecretClient
	endpoint   string
	// extraHeaders are sent on every request (e.g. X-Custom-Auth).
	extraHeaders map[string]string
	// Endpoint path overrides. Empty means the provider default.
	chatCompletionsPath string
	responsesPath        string
	imagesPath           string
	embeddingsPath       string
}

// Config groups construction inputs.
type Config struct {
	HTTPClient *http.Client          // optional; defaults to pooled client w/ 300s timeout
	Secrets    domain.SecretClient   // required (BYOA resolution)
	Endpoint   string                // optional override (vLLM / test stub)
	ExtraHeaders map[string]string   // optional extra headers on every request
	ChatCompletionsPath string       // optional override for /chat/completions
	ResponsesPath        string       // optional override for /responses
	ImagesPath           string       // optional override for /images/generations
	EmbeddingsPath       string       // optional override for /embeddings
}

// New constructs a Client. Required-field validation is fail-loud.
func New(cfg Config) (*Client, error) {
	if cfg.Secrets == nil {
		return nil, fmt.Errorf("openai: SecretClient required")
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = newPooledClient()
	}
	endpoint := cfg.Endpoint
	if endpoint == "" {
		endpoint = defaultEndpoint
	}
	return &Client{
		httpClient:           hc,
		secrets:              cfg.Secrets,
		endpoint:             endpoint,
		extraHeaders:         cfg.ExtraHeaders,
		chatCompletionsPath:  cfg.ChatCompletionsPath,
		responsesPath:        cfg.ResponsesPath,
		imagesPath:           cfg.ImagesPath,
		embeddingsPath:       cfg.EmbeddingsPath,
	}, nil
}

// newPooledClient returns an *http.Client with connection pooling and a
// layered timeout hierarchy:
//
//   - Connection timeout (10s): bounds TCP dial + TLS handshake. A provider
//     that accepts the TCP connection but never completes the handshake is
//     cut off here.
//   - Response-header timeout (30s): bounds the wait for the first response
//     byte after the request is sent. A provider that accepts the request
//     but never responds is cut off here.
//   - Overall provider timeout (300s): bounds the entire request/response
//     cycle. A grounded reasoning run makes several upstream calls and can
//     take well over a minute, so this is generous.
//
// The overall timeout is enforced by the http.Client.Timeout field; the
// connection and response-header timeouts are enforced by the Transport.
// The caller's context (which carries the gateway deadline) is the outer
// bound — a client disconnect cancels the request regardless of these
// timeouts.
func newPooledClient() *http.Client {
	return &http.Client{
		Timeout: 300 * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:          100,
			MaxIdleConnsPerHost:   10,
			IdleConnTimeout:       90 * time.Second,
			TLSHandshakeTimeout:   10 * time.Second,
			ResponseHeaderTimeout: 30 * time.Second,
			ExpectContinueTimeout: 1 * time.Second,
			DialContext: (&net.Dialer{
				Timeout: 10 * time.Second,
			}).DialContext,
		},
	}
}

// Family — implements domain.VendorClient.
func (c *Client) Family() domain.VendorFamily { return domain.VendorFamilyOpenAI }

// Generate — implements domain.VendorClient.
//
// Dispatches to the responses API when the request is grounded (carries a
// web_search tool in ToolsJSON), otherwise to chat completions. The
// responses API is OpenAI's hosted web-search surface; chat completions
// is the plain text/tool-call surface.
func (c *Client) Generate(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	if err := domain.CheckContextCancellation(ctx); err != nil {
		return domain.VendorResponse{}, err
	}
	apiKey, err := c.resolveKey(ctx, req.TenantID)
	if err != nil {
		return domain.VendorResponse{}, err
	}

	if isGrounded(req) {
		return c.generateResponses(ctx, apiKey, req)
	}
	return c.generateChat(ctx, apiKey, req)
}

// GenerateImage generates an image via the OpenAI images/generations
// endpoint. Implements the image half of the vendor surface.
func (c *Client) GenerateImage(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	if err := domain.CheckContextCancellation(ctx); err != nil {
		return domain.VendorResponse{}, err
	}
	if strings.TrimSpace(req.Prompt) == "" {
		return domain.VendorResponse{}, fmt.Errorf("openai: image generation requires a prompt")
	}
	apiKey, err := c.resolveKey(ctx, req.TenantID)
	if err != nil {
		return domain.VendorResponse{}, err
	}

	body := buildImageBody(req)
	rawBody, err := json.Marshal(body)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: marshal image request: %w", err)
	}

	url := c.resolveEndpoint(c.imagesPath, "/images/generations")
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(rawBody))
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: build image request: %w", err)
	}
	c.setHeaders(httpReq, apiKey)

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: image HTTP do: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		return domain.VendorResponse{}, &ProviderError{
			Status:   httpResp.StatusCode,
			Endpoint: url,
			Body:     truncate(string(respBody), 2048),
		}
	}
	var parsed imageResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: unmarshal image response: %w", err)
	}
	return parsed.toDomain(string(req.LogicalModelID), c.endpoint, c.imagesPath), nil
}

// EmbedText produces one text embedding via the OpenAI embeddings
// endpoint. Implements domain.EmbeddingClient.
func (c *Client) EmbedText(ctx context.Context, req domain.EmbedVendorRequest) (domain.EmbedVendorResponse, error) {
	if err := domain.CheckContextCancellation(ctx); err != nil {
		return domain.EmbedVendorResponse{}, err
	}
	apiKey, err := c.resolveKey(ctx, req.TenantID)
	if err != nil {
		return domain.EmbedVendorResponse{}, err
	}

	body := buildEmbeddingsBody(req)
	rawBody, err := json.Marshal(body)
	if err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("openai: marshal embeddings request: %w", err)
	}

	url := c.resolveEndpoint(c.embeddingsPath, "/embeddings")
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(rawBody))
	if err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("openai: build embeddings request: %w", err)
	}
	c.setHeaders(httpReq, apiKey)

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("openai: embeddings HTTP do: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		return domain.EmbedVendorResponse{}, &ProviderError{
			Status:   httpResp.StatusCode,
			Endpoint: url,
			Body:     truncate(string(respBody), 2048),
		}
	}
	var parsed embeddingsResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("openai: unmarshal embeddings response: %w", err)
	}
	return parsed.toDomain(string(req.LogicalModelID)), nil
}

// ---------------------------------------------------------------------------
// Key resolution
// ---------------------------------------------------------------------------

func (c *Client) resolveKey(ctx context.Context, tenantID string) (string, error) {
	apiKey, err := c.secrets.ResolveByoaKey(ctx, tenantID, domain.VendorFamilyOpenAI)
	if err != nil {
		return "", fmt.Errorf("openai: resolve BYOA: %w", err)
	}
	if apiKey == "" {
		apiKey, err = c.secrets.ResolveGlobalKey(ctx, domain.VendorFamilyOpenAI)
		if err != nil {
			return "", fmt.Errorf("openai: resolve global key: %w", err)
		}
	}
	if apiKey == "" {
		return "", fmt.Errorf("openai: no API key (BYOA + global both empty)")
	}
	return apiKey, nil
}

// ---------------------------------------------------------------------------
// Header + endpoint helpers
// ---------------------------------------------------------------------------

func (c *Client) setHeaders(httpReq *http.Request, apiKey string) {
	for k, v := range c.extraHeaders {
		if domain.IsGatewayControlledHeader(k) {
			continue
		}
		httpReq.Header.Set(k, v)
	}
	httpReq.Header.Set("Authorization", "Bearer "+apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
}

// resolveEndpoint builds the full URL for an API call, honoring a path
// override. An absolute http(s) URL is used verbatim; a relative path is
// joined onto the base endpoint.
func (c *Client) resolveEndpoint(path, defaultPath string) string {
	if path == "" {
		path = defaultPath
	}
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		return path
	}
	return strings.TrimRight(c.endpoint, "/") + "/" + strings.TrimLeft(path, "/")
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

func (c *Client) generateChat(ctx context.Context, apiKey string, req domain.VendorRequest) (domain.VendorResponse, error) {
	if err := domain.CheckContextCancellation(ctx); err != nil {
		return domain.VendorResponse{}, err
	}
	body := buildOpenAIBody(req)
	rawBody, err := json.Marshal(body)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: marshal request: %w", err)
	}

	url := c.resolveEndpoint(c.chatCompletionsPath, "/v1/chat/completions")
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(rawBody))
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: build request: %w", err)
	}
	c.setHeaders(httpReq, apiKey)

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: HTTP do: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		return domain.VendorResponse{}, &ProviderError{
			Status:   httpResp.StatusCode,
			Endpoint: url,
			Body:     truncate(string(respBody), 2048),
		}
	}
	var parsed openaiResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: unmarshal response: %w", err)
	}
	return parsed.toDomain(string(req.LogicalModelID)), nil
}

// ---------------------------------------------------------------------------
// Responses API (hosted web search)
// ---------------------------------------------------------------------------

func (c *Client) generateResponses(ctx context.Context, apiKey string, req domain.VendorRequest) (domain.VendorResponse, error) {
	if err := domain.CheckContextCancellation(ctx); err != nil {
		return domain.VendorResponse{}, err
	}
	body := buildResponsesBody(req)
	rawBody, err := json.Marshal(body)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: marshal responses request: %w", err)
	}

	url := c.resolveEndpoint(c.responsesPath, "/v1/responses")
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(rawBody))
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: build responses request: %w", err)
	}
	c.setHeaders(httpReq, apiKey)

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: responses HTTP do: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		return domain.VendorResponse{}, &ProviderError{
			Status:   httpResp.StatusCode,
			Endpoint: url,
			Body:     truncate(string(respBody), 2048),
		}
	}
	var parsed responsesResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return domain.VendorResponse{}, fmt.Errorf("openai: unmarshal responses: %w", err)
	}
	return parsed.toDomain(string(req.LogicalModelID)), nil
}

// ---------------------------------------------------------------------------
// Wire shapes — chat completions
// ---------------------------------------------------------------------------

type openaiBody struct {
	Model       string            `json:"model"`
	Messages    []openaiMessage   `json:"messages"`
	Stream      bool              `json:"stream"`
	Temperature *float64          `json:"temperature,omitempty"`
	MaxTokens   *int              `json:"max_tokens,omitempty"`
	TopP        *float64          `json:"top_p,omitempty"`
	N           *int              `json:"n,omitempty"`
	Seed        *int              `json:"seed,omitempty"`
	Stop        []string          `json:"stop,omitempty"`
	ToolChoice  json.RawMessage   `json:"tool_choice,omitempty"`
	ResponseFormat json.RawMessage `json:"response_format,omitempty"`
	Tools       json.RawMessage   `json:"tools,omitempty"`
}

type openaiMessage struct {
	Role        string              `json:"role"`
	Content     string              `json:"content"`
	Annotations []openaiAnnotation  `json:"annotations,omitempty"`
	ToolCalls   []openaiToolCall    `json:"tool_calls,omitempty"`
}

type openaiAnnotation struct {
	Type       string `json:"type"`
	URL        string `json:"url"`
	Title      string `json:"title"`
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
}

type openaiToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openaiResponse struct {
	Model   string `json:"model"`
	Choices []struct {
		Message      openaiMessage `json:"message"`
		FinishReason string        `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
	} `json:"usage"`
}

func buildOpenAIBody(req domain.VendorRequest) openaiBody {
	body := openaiBody{
		Model:  string(req.LogicalModelID),
		Stream: false,
	}
	// Build messages from ContentsJSON if present, else flat prompt.
	messages := buildChatMessages(req)
	body.Messages = messages
	// Tool declarations — raw passthrough.
	if strings.TrimSpace(req.ToolsJSON) != "" {
		body.Tools = json.RawMessage(req.ToolsJSON)
	}
	applyGenerationConfig(&body, req.GenerationConfig)
	return body
}

func buildChatMessages(req domain.VendorRequest) []openaiMessage {
	if strings.TrimSpace(req.ContentsJSON) != "" {
		var contents []openaiMessage
		if err := json.Unmarshal([]byte(req.ContentsJSON), &contents); err == nil && len(contents) > 0 {
			return contents
		}
	}
	messages := []openaiMessage{}
	if req.SystemPrompt != "" {
		messages = append(messages, openaiMessage{Role: "system", Content: req.SystemPrompt})
	}
	messages = append(messages, openaiMessage{Role: "user", Content: req.Prompt})
	return messages
}

// applyGenerationConfig projects the vendor-neutral generation parameters
// onto the OpenAI chat body. Supports: temperature, top_p, max_tokens,
// max_completion_tokens, n, seed, stop, tool_choice, response_format.
func applyGenerationConfig(body *openaiBody, cfg map[string]any) {
	for key, value := range cfg {
		if value == nil {
			continue
		}
		switch key {
		case "temperature", "top_p":
			if n := toFloat(value); n != nil {
				if key == "temperature" {
					body.Temperature = n
				} else {
					body.TopP = n
				}
			}
		case "max_tokens", "max_completion_tokens":
			if n := toFloat(value); n != nil {
				mt := int(*n)
				body.MaxTokens = &mt
			}
		case "n":
			if n := toFloat(value); n != nil {
				nn := int(*n)
				body.N = &nn
			}
		case "seed":
			if n := toFloat(value); n != nil {
				s := int(*n)
				body.Seed = &s
			}
		case "stop":
			switch v := value.(type) {
			case string:
				body.Stop = []string{v}
			case []string:
				body.Stop = v
			case []any:
				var stops []string
				for _, item := range v {
					if s, ok := item.(string); ok {
						stops = append(stops, s)
					}
				}
				if len(stops) > 0 {
					body.Stop = stops
				}
			}
		case "tool_choice":
			raw, _ := json.Marshal(value)
			body.ToolChoice = raw
		case "response_format":
			raw, _ := json.Marshal(value)
			body.ResponseFormat = raw
		}
	}
}

func (r openaiResponse) toDomain(fallbackModel string) domain.VendorResponse {
	resp := domain.VendorResponse{
		Usage: domain.TokenUsage{
			InputTokens:  r.Usage.PromptTokens,
			OutputTokens: r.Usage.CompletionTokens,
			CostMicros:   openaiCostMicros(fallbackModel, r.Usage.PromptTokens, r.Usage.CompletionTokens),
		},
		ModelVersion: r.Model,
	}
	if resp.ModelVersion == "" {
		resp.ModelVersion = fallbackModel
	}
	if len(r.Choices) > 0 {
		choice := r.Choices[0]
		resp.Completion = choice.Message.Content
		resp.FinishReason = mapOpenAIFinishReason(choice.FinishReason)
		resp.Citations = extractAnnotations(choice.Message.Annotations)
		// Tool calls (ADR-177).
		if len(choice.Message.ToolCalls) > 0 {
			tc, _ := json.Marshal(choice.Message.ToolCalls)
			resp.ToolCallsJSON = string(tc)
		}
	}
	return resp
}

func mapOpenAIFinishReason(s string) domain.FinishReason {
	switch s {
	case "stop", "tool_calls", "function_call":
		return domain.FinishReasonComplete
	case "length":
		return domain.FinishReasonMaxTokens
	case "content_filter":
		return domain.FinishReasonModelArmorBlock
	default:
		return domain.FinishReasonUnspecified
	}
}

// ---------------------------------------------------------------------------
// Wire shapes — responses API
// ---------------------------------------------------------------------------

type responsesBody struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input,omitempty"`
	Instructions   string          `json:"instructions,omitempty"`
	Stream         bool            `json:"stream"`
	Tools          json.RawMessage `json:"tools,omitempty"`
	TopP           *float64        `json:"top_p,omitempty"`
	MaxOutputTokens *int           `json:"max_output_tokens,omitempty"`
	Seed           *int            `json:"seed,omitempty"`
}

type responsesResponse struct {
	Model     string               `json:"model"`
	Status    string               `json:"status"`
	Output    []responsesOutputItem `json:"output"`
	Usage     responsesUsage       `json:"usage"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

type responsesOutputItem struct {
	Type    string                `json:"type"`
	Content []responsesContentPart `json:"content"`
	Action  *responsesAction      `json:"action"`
}

type responsesContentPart struct {
	Type        string                `json:"type"`
	Text        string                `json:"text"`
	Annotations []responsesAnnotation `json:"annotations"`
}

type responsesAnnotation struct {
	Type      string `json:"type"`
	URL       string `json:"url"`
	Title     string `json:"title"`
	StartIndex int   `json:"start_index"`
	EndIndex   int   `json:"end_index"`
}

type responsesAction struct {
	Type    string   `json:"type"`
	Query   string   `json:"query"`
	Queries []string `json:"queries"`
}

type responsesUsage struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	InputTokensDetails *struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
}

func buildResponsesBody(req domain.VendorRequest) responsesBody {
	body := responsesBody{
		Model:  string(req.LogicalModelID),
		Stream: false,
	}
	// Input: ContentsJSON if present, else flat prompt.
	if strings.TrimSpace(req.ContentsJSON) != "" {
		body.Input = json.RawMessage(req.ContentsJSON)
	} else {
		body.Input = json.RawMessage(fmt.Sprintf("%q", req.Prompt))
	}
	if req.SystemPrompt != "" {
		body.Instructions = req.SystemPrompt
	}
	// Tools: passthrough + web_search if grounded.
	if strings.TrimSpace(req.ToolsJSON) != "" {
		body.Tools = json.RawMessage(req.ToolsJSON)
	}
	// Apply generation config (responses API subset).
	for key, value := range req.GenerationConfig {
		if value == nil {
			continue
		}
		switch key {
		case "temperature", "top_p":
			if n := toFloat(value); n != nil {
				body.TopP = n
			}
		case "max_output_tokens", "max_tokens":
			if n := toFloat(value); n != nil {
				mt := int(*n)
				body.MaxOutputTokens = &mt
			}
		case "seed":
			if n := toFloat(value); n != nil {
				s := int(*n)
				body.Seed = &s
			}
		}
	}
	return body
}

func (r responsesResponse) toDomain(fallbackModel string) domain.VendorResponse {
	resp := domain.VendorResponse{
		Usage: domain.TokenUsage{
			InputTokens:  r.Usage.InputTokens,
			OutputTokens: r.Usage.OutputTokens,
			CostMicros:   openaiCostMicros(fallbackModel, r.Usage.InputTokens, r.Usage.OutputTokens),
		},
		ModelVersion: r.Model,
	}
	if resp.ModelVersion == "" {
		resp.ModelVersion = fallbackModel
	}

	// Status handling.
	switch r.Status {
	case "", "completed":
		resp.FinishReason = domain.FinishReasonComplete
	case "incomplete":
		reason := "the provider stopped before finishing"
		if r.IncompleteDetails != nil && r.IncompleteDetails.Reason != "" {
			reason = r.IncompleteDetails.Reason
		}
		resp.FinishReason = domain.FinishReasonMaxTokens
		resp.FinishDetail = fmt.Sprintf(
			"the response was truncated (%s); a grounded run needs enough max_output_tokens to finish its searches",
			reason,
		)
	case "failed":
		resp.FinishReason = domain.FinishReasonVendorError
		msg := "the provider reported a failure"
		if r.Error != nil && r.Error.Message != "" {
			msg = r.Error.Message
		}
		resp.FinishDetail = "the provider reported a failure: " + msg
	case "cancelled":
		resp.FinishReason = domain.FinishReasonVendorError
		resp.FinishDetail = "the response was cancelled upstream"
	default:
		resp.FinishReason = domain.FinishReasonUnspecified
		resp.FinishDetail = "the response finished in status " + r.Status
	}

	// Output items: web_search_call + message.
	for _, item := range r.Output {
		switch item.Type {
		case "web_search_call":
			resp.SearchQueries = append(resp.SearchQueries, extractSearchQueries(item)...)
		case "message":
			var text strings.Builder
			var citations []domain.Citation
			for _, part := range item.Content {
				if part.Type == "output_text" || part.Type == "text" || part.Type == "" {
					text.WriteString(part.Text)
				}
				citations = append(citations, extractResponsesAnnotations(part.Annotations)...)
			}
			t := text.String()
			if t != "" || len(citations) > 0 {
				resp.Citations = append(resp.Citations, citations...)
			}
			if t != "" {
				resp.Completion = t
			}
		}
	}

	// If complete but no text, mark unspecified.
	if resp.FinishReason == domain.FinishReasonComplete && resp.Completion == "" {
		resp.FinishReason = domain.FinishReasonUnspecified
		resp.FinishDetail = "the provider returned no output_text content"
	}
	return resp
}

// ---------------------------------------------------------------------------
// Wire shapes — images
// ---------------------------------------------------------------------------

type imageBody struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	N      int    `json:"n"`
}

type imageResponse struct {
	Data []struct {
		B64JSON       string `json:"b64_json"`
		URL           string `json:"url"`
		RevisedPrompt string `json:"revised_prompt"`
	} `json:"data"`
	Usage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
	} `json:"usage"`
}

func buildImageBody(req domain.VendorRequest) imageBody {
	return imageBody{
		Model:  string(req.LogicalModelID),
		Prompt: req.Prompt,
		N:      1,
	}
}

func (r imageResponse) toDomain(fallbackModel, endpoint, imagesPath string) domain.VendorResponse {
	resp := domain.VendorResponse{
		Usage: domain.TokenUsage{
			InputTokens:  r.Usage.PromptTokens,
			OutputTokens: r.Usage.CompletionTokens,
			CostMicros:   0, // images are unpriced in the current table
		},
		ModelVersion: fallbackModel,
		FinishReason: domain.FinishReasonComplete,
	}
	if len(r.Data) == 0 {
		resp.FinishReason = domain.FinishReasonUnspecified
		resp.FinishDetail = "upstream returned no image data"
		return resp
	}
	first := r.Data[0]
	resp.RevisedPrompt = first.RevisedPrompt
	if first.B64JSON != "" {
		raw, err := base64.StdEncoding.DecodeString(first.B64JSON)
		if err != nil {
			resp.FinishReason = domain.FinishReasonVendorError
			resp.FinishDetail = fmt.Sprintf("decode image b64_json: %v", err)
			return resp
		}
		resp.ImageBytes = raw
		resp.ImageMIMEType = SniffImageMime(raw)
	} else if first.URL != "" {
		raw, mime, err := fetchImage(context.Background(), first.URL)
		if err != nil {
			resp.FinishReason = domain.FinishReasonVendorError
			resp.FinishDetail = fmt.Sprintf("fetch image URL: %v", err)
			return resp
		}
		resp.ImageBytes = raw
		resp.ImageMIMEType = mime
	} else {
		resp.FinishReason = domain.FinishReasonVendorError
		resp.FinishDetail = "image response carried neither b64_json nor url"
		return resp
	}
	return resp
}

// ---------------------------------------------------------------------------
// Wire shapes — embeddings
// ---------------------------------------------------------------------------

type embeddingsBody struct {
	Model      string   `json:"model"`
	Input      string   `json:"input"`
	Dimensions *int     `json:"dimensions,omitempty"`
}

type embeddingsResponse struct {
	Model string `json:"model"`
	Data  []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Usage struct {
		PromptTokens int64 `json:"prompt_tokens"`
	} `json:"usage"`
}

func buildEmbeddingsBody(req domain.EmbedVendorRequest) embeddingsBody {
	body := embeddingsBody{
		Model: string(req.LogicalModelID),
		Input: req.Text,
	}
	if req.OutputDimensions > 0 {
		d := int(req.OutputDimensions)
		body.Dimensions = &d
	}
	return body
}

func (r embeddingsResponse) toDomain(fallbackModel string) domain.EmbedVendorResponse {
	model := r.Model
	if model == "" {
		model = fallbackModel
	}
	if len(r.Data) == 0 {
		return domain.EmbedVendorResponse{ModelVersion: model}
	}
	return domain.EmbedVendorResponse{
		Values:       r.Data[0].Embedding,
		ModelVersion: model,
		InputTokens:  r.Usage.PromptTokens,
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// isGrounded reports whether the request carries a web_search tool.
func isGrounded(req domain.VendorRequest) bool {
	if strings.TrimSpace(req.ToolsJSON) == "" {
		return false
	}
	var tools []map[string]any
	if err := json.Unmarshal([]byte(req.ToolsJSON), &tools); err != nil {
		return false
	}
	for _, tool := range tools {
		for _, key := range []string{"type", "name"} {
			if v, ok := tool[key]; ok {
				s := strings.ToLower(fmt.Sprintf("%v", v))
				if strings.Contains(s, "web_search") || strings.Contains(s, "web_fetch") {
					return true
				}
			}
		}
	}
	return false
}

// extractAnnotations extracts URL citations from OpenAI chat message
// annotations.
func extractAnnotations(annotations []openaiAnnotation) []domain.Citation {
	var out []domain.Citation
	for _, a := range annotations {
		if a.URL == "" {
			continue
		}
		if a.Type != "" && a.Type != "url_citation" {
			continue
		}
		out = append(out, domain.Citation{
			URL:        a.URL,
			Title:      a.Title,
			StartIndex: a.StartIndex,
			EndIndex:   a.EndIndex,
		})
	}
	return out
}

// extractResponsesAnnotations extracts URL citations from responses API
// content part annotations.
func extractResponsesAnnotations(annotations []responsesAnnotation) []domain.Citation {
	var out []domain.Citation
	for _, a := range annotations {
		if a.URL == "" {
			continue
		}
		if a.Type != "" && a.Type != "url_citation" {
			continue
		}
		out = append(out, domain.Citation{
			URL:        a.URL,
			Title:      a.Title,
			StartIndex: a.StartIndex,
			EndIndex:   a.EndIndex,
		})
	}
	return out
}

// extractSearchQueries extracts search queries from a web_search_call
// output item.
func extractSearchQueries(item responsesOutputItem) []string {
	if item.Action == nil {
		return nil
	}
	action := item.Action
	if action.Type != "" && action.Type != "search" {
		return nil
	}
	var queries []string
	if action.Query != "" {
		queries = append(queries, action.Query)
	}
	queries = append(queries, action.Queries...)
	return queries
}

// SniffImageMime sniffs the MIME type from magic bytes.
func SniffImageMime(data []byte) string {
	if len(data) >= 8 && string(data[:8]) == "\x89PNG\r\n\x1a\n" {
		return "image/png"
	}
	if len(data) >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
		return "image/jpeg"
	}
	if len(data) >= 12 && string(data[:4]) == "RIFF" && string(data[8:12]) == "WEBP" {
		return "image/webp"
	}
	if len(data) >= 6 && (string(data[:6]) == "GIF87a" || string(data[:6]) == "GIF89a") {
		return "image/gif"
	}
	return "image/png"
}

// fetchImage fetches an image from a URL with a 32 MiB limit. The context is
// honoured so a client disconnect cancels the fetch.
func fetchImage(ctx context.Context, url string) ([]byte, string, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, "", fmt.Errorf("fetch image: %w", err)
	}
	resp, err := http.DefaultClient.Do(httpReq) //nolint:gosec,noctx // provider-minted pre-signed URL
	if err != nil {
		return nil, "", fmt.Errorf("fetch image: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode >= 400 {
		return nil, "", &ProviderError{
			Status:   resp.StatusCode,
			Endpoint: url,
			Body:     fmt.Sprintf("image fetch failed: %d", resp.StatusCode),
		}
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, maxImageFetchBytes))
	if err != nil {
		return nil, "", fmt.Errorf("read image body: %w", err)
	}
	mime := resp.Header.Get("Content-Type")
	if mime == "" {
		mime = SniffImageMime(data)
	}
	return data, mime, nil
}

// toFloat converts any to a float64, returning nil for non-numeric types
// and bools.
func toFloat(v any) *float64 {
	switch n := v.(type) {
	case float64:
		return &n
	case float32:
		f := float64(n)
		return &f
	case int:
		f := float64(n)
		return &f
	case int32:
		f := float64(n)
		return &f
	case int64:
		f := float64(n)
		return &f
	default:
		return nil
	}
}

// truncate truncates a string to limit bytes, appending an ellipsis.
func truncate(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return s[:limit] + "…"
}

// ---------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------

// openaiPricing — micro-USD per 1M tokens. Hard-coded for Phase 2.2;
// Phase 4 follow-up moves this into GCS-managed YAML config.
var openaiPricing = map[string]struct {
	InputMicrosPer1M  int64
	OutputMicrosPer1M int64
}{
	"gpt-4o":            {2_500_000, 10_000_000},
	"gpt-4o-mini":       {150_000, 600_000},
	"gpt-4.1":           {2_000_000, 8_000_000},
	"gpt-4.1-mini":      {400_000, 1_600_000},
	// Gemma-via-vLLM on self-hosted infra is the platform-controlled path;
	// platform attributes 0 vendor cost because Gemma's compute cost is
	// captured separately in the GPU-utilisation cost ledger (Phase 4).
	"gemma-sg-academic": {0, 0},
}

func openaiCostMicros(model string, inputTokens, outputTokens int64) int64 {
	p, ok := openaiPricing[model]
	if !ok {
		return 0
	}
	in := (inputTokens * p.InputMicrosPer1M) / 1_000_000
	out := (outputTokens * p.OutputMicrosPer1M) / 1_000_000
	return in + out
}

// Compile-time port guarantees.
var _ domain.VendorClient = (*Client)(nil)
var _ domain.EmbeddingClient = (*Client)(nil)
