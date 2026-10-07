package openai_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/adapter/vendors/openai"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubSecrets struct {
	byoa   string
	byoaErr error
	global string
	globalErr error
}

func (s stubSecrets) ResolveByoaKey(ctx context.Context, tenantID string, vendor domain.VendorFamily) (string, error) {
	return s.byoa, s.byoaErr
}
func (s stubSecrets) ResolveGlobalKey(ctx context.Context, vendor domain.VendorFamily) (string, error) {
	return s.global, s.globalErr
}

func TestOpenAI_Family(t *testing.T) {
	c, err := openai.New(openai.Config{Secrets: stubSecrets{}})
	require.NoError(t, err)
	assert.Equal(t, domain.VendorFamilyOpenAI, c.Family())
}

func TestOpenAI_GenerateHappyPath_BYOAKey(t *testing.T) {
	var capturedAuth string
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{
			"model": "gpt-4o-mini-2025-07-18",
			"choices": [{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
			"usage": {"prompt_tokens": 50, "completion_tokens": 20}
		}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{
		HTTPClient: srv.Client(),
		Secrets:    stubSecrets{byoa: "sk-tenant-byoa"},
		Endpoint:   srv.URL,
	})
	require.NoError(t, err)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:         "tenant-1",
		LogicalModelID:   "gpt-4o-mini",
		Prompt:           "ping",
		SystemPrompt:     "you reply pong",
		GenerationConfig: map[string]any{"temperature": 0.5, "max_tokens": 64},
	})
	require.NoError(t, err)

	assert.Equal(t, "Bearer sk-tenant-byoa", capturedAuth)
	assert.Equal(t, "gpt-4o-mini", capturedBody["model"])
	require.Len(t, capturedBody["messages"], 2)
	first := capturedBody["messages"].([]any)[0].(map[string]any)
	assert.Equal(t, "system", first["role"])
	assert.Equal(t, 0.5, capturedBody["temperature"])

	assert.Equal(t, "hello", resp.Completion)
	assert.Equal(t, "gpt-4o-mini-2025-07-18", resp.ModelVersion)
	assert.Equal(t, int64(50), resp.Usage.InputTokens)
	assert.Equal(t, int64(20), resp.Usage.OutputTokens)
	// Pricing is micro-USD PER 1M tokens (gpt-4o-mini: 150_000 input + 600_000 output).
	// Cost = 50 * 150_000 / 1_000_000 + 20 * 600_000 / 1_000_000
	//      = 7 (truncated from 7.5) + 12 = 19 micro-USD.
	assert.Equal(t, int64(19), resp.Usage.CostMicros)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
}

func TestOpenAI_GenerateGlobalKeyFallback(t *testing.T) {
	var capturedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedAuth = r.Header.Get("Authorization")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{
		HTTPClient: srv.Client(),
		Secrets:    stubSecrets{byoa: "", global: "sk-global-fallback"},
		Endpoint:   srv.URL,
	})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.NoError(t, err)
	assert.Equal(t, "Bearer sk-global-fallback", capturedAuth)
}

func TestOpenAI_GenerateNoKey(t *testing.T) {
	c, err := openai.New(openai.Config{Secrets: stubSecrets{}})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no API key")
}

func TestOpenAI_GenerateBYOAError(t *testing.T) {
	c, err := openai.New(openai.Config{Secrets: stubSecrets{byoaErr: errors.New("SM permission denied")}})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "BYOA")
}

func TestOpenAI_GenerateHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limit"}}`))
	}))
	defer srv.Close()
	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "429")
}

func TestOpenAI_GenerateLengthFinish(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"length"}],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonMaxTokens, resp.FinishReason)
}

func TestOpenAI_GenerateContentFilter(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":""},"finish_reason":"content_filter"}],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
}

func TestOpenAI_NewValidation(t *testing.T) {
	_, err := openai.New(openai.Config{Secrets: nil})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "SecretClient")
}

func TestOpenAI_GenerateUnknownModel_ZeroCost(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}`))
	}))
	defer srv.Close()
	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "vapor-model-x"})
	require.NoError(t, err)
	assert.Equal(t, int64(0), resp.Usage.CostMicros)
}

func TestOpenAI_GenerateResponses_Grounded(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{
			"model": "gpt-4o-2025-08-06",
			"status": "completed",
			"output": [
				{"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "most recent F1 race winner"}},
				{"type": "message", "id": "msg_1", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "Max Verstappen won.", "annotations": [{"type": "url_citation", "url": "https://f1.example/race-report", "title": "Race report", "start_index": 0, "end_index": 3}]}]}
			],
			"usage": {"input_tokens": 25, "output_tokens": 12, "input_tokens_details": {"cached_tokens": 0}}
		}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "gpt-4o",
		Prompt:         "who won the most recent F1 race?",
		ToolsJSON:      `[{"type": "web_search"}]`,
	})
	require.NoError(t, err)

	assert.Equal(t, false, capturedBody["stream"])
	assert.Equal(t, "gpt-4o", capturedBody["model"])
	assert.Equal(t, "who won the most recent F1 race?", capturedBody["input"])
	assert.NotNil(t, capturedBody["tools"])
	assert.Equal(t, "Max Verstappen won.", resp.Completion)
	assert.Equal(t, "gpt-4o-2025-08-06", resp.ModelVersion)
	assert.Equal(t, int64(25), resp.Usage.InputTokens)
	assert.Equal(t, int64(12), resp.Usage.OutputTokens)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
	assert.Len(t, resp.Citations, 1)
	assert.Equal(t, "https://f1.example/race-report", resp.Citations[0].URL)
	assert.Equal(t, "Race report", resp.Citations[0].Title)
	assert.Equal(t, 0, resp.Citations[0].StartIndex)
	assert.Equal(t, 3, resp.Citations[0].EndIndex)
	assert.Equal(t, []string{"most recent F1 race winner"}, resp.SearchQueries)
}

func TestOpenAI_GenerateResponses_IncompleteStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"model": "gpt-4o-2025-08-06",
			"status": "incomplete",
			"incomplete_details": {"reason": "max_output_tokens"},
			"output": [{"type": "message", "id": "m", "status": "in_progress", "role": "assistant", "content": [{"type": "output_text", "text": "half an answer"}]}],
			"usage": {"input_tokens": 10, "output_tokens": 5, "input_tokens_details": {"cached_tokens": 0}}
		}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "gpt-4o",
		Prompt:         "who won?",
		ToolsJSON:      `[{"type": "web_search"}]`,
	})
	require.NoError(t, err)

	assert.Equal(t, domain.FinishReasonMaxTokens, resp.FinishReason)
	assert.Contains(t, resp.FinishDetail, "truncated")
	assert.Contains(t, resp.FinishDetail, "max_output_tokens")
}

func TestOpenAI_GenerateResponses_FailedStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"model": "gpt-4o-2025-08-06",
			"status": "failed",
			"error": {"message": "internal server error"},
			"output": [],
			"usage": {"input_tokens": 10, "output_tokens": 0}
		}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "gpt-4o",
		Prompt:         "who won?",
		ToolsJSON:      `[{"type": "web_search"}]`,
	})
	require.NoError(t, err)

	assert.Equal(t, domain.FinishReasonVendorError, resp.FinishReason)
	assert.Contains(t, resp.FinishDetail, "internal server error")
}

func TestOpenAI_GenerateResponses_CancelledStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"model": "gpt-4o-2025-08-06",
			"status": "cancelled",
			"output": [],
			"usage": {"input_tokens": 10, "output_tokens": 0}
		}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "gpt-4o",
		Prompt:         "who won?",
		ToolsJSON:      `[{"type": "web_search"}]`,
	})
	require.NoError(t, err)

	assert.Equal(t, domain.FinishReasonVendorError, resp.FinishReason)
	assert.Contains(t, resp.FinishDetail, "cancelled")
}

func TestOpenAI_GenerateResponses_NoOutputText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"model": "gpt-4o-2025-08-06",
			"status": "completed",
			"output": [{"type": "web_search_call", "id": "ws_1", "status": "completed", "action": {"type": "search", "query": "q"}}],
			"usage": {"input_tokens": 10, "output_tokens": 0}
		}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "gpt-4o",
		Prompt:         "who won?",
		ToolsJSON:      `[{"type": "web_search"}]`,
	})
	require.NoError(t, err)

	assert.Equal(t, domain.FinishReasonUnspecified, resp.FinishReason)
	assert.Contains(t, resp.FinishDetail, "no output_text")
}

func TestOpenAI_GenerateChat_CitationsFromAnnotations(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"model": "gpt-4o-2025-08-06",
			"choices": [{"message": {"role": "assistant", "content": "hello", "annotations": [{"type": "url_citation", "url": "https://example.com", "title": "Example", "start_index": 0, "end_index": 5}]}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "gpt-4o",
		Prompt:         "hi",
	})
	require.NoError(t, err)

	assert.Len(t, resp.Citations, 1)
	assert.Equal(t, "https://example.com", resp.Citations[0].URL)
	assert.Equal(t, "Example", resp.Citations[0].Title)
	assert.Equal(t, 0, resp.Citations[0].StartIndex)
	assert.Equal(t, 5, resp.Citations[0].EndIndex)
}

func TestOpenAI_GenerateChat_ToolCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"model": "gpt-4o-2025-08-06",
			"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}],
			"usage": {"prompt_tokens": 10, "output_tokens": 5}
		}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "gpt-4o",
		Prompt:         "look up X",
	})
	require.NoError(t, err)

	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
	assert.NotEmpty(t, resp.ToolCallsJSON)
}

func TestOpenAI_GenerateChat_GenerationConfigBreadth(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	_, err = c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "gpt-4o",
		Prompt:         "hi",
		GenerationConfig: map[string]any{
			"temperature":      0.7,
			"top_p":            0.9,
			"max_tokens":       100,
			"n":                2,
			"seed":             42,
			"stop":             []string{"END"},
			"tool_choice":      "auto",
			"response_format":  map[string]string{"type": "json_object"},
		},
	})
	require.NoError(t, err)

	assert.Equal(t, 0.7, capturedBody["temperature"])
	assert.Equal(t, 0.9, capturedBody["top_p"])
	assert.Equal(t, float64(100), capturedBody["max_tokens"])
	assert.Equal(t, float64(2), capturedBody["n"])
	assert.Equal(t, float64(42), capturedBody["seed"])
	assert.Equal(t, []any{"END"}, capturedBody["stop"])
	assert.Equal(t, "auto", capturedBody["tool_choice"])
	assert.Equal(t, map[string]any{"type": "json_object"}, capturedBody["response_format"])
}

func TestOpenAI_GenerateChat_ExtraHeaders(t *testing.T) {
	var capturedHeaders http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeaders = r.Header.Clone()
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{
		HTTPClient:   srv.Client(),
		Secrets:      stubSecrets{byoa: "k"},
		Endpoint:     srv.URL,
		ExtraHeaders: map[string]string{"X-Custom-Auth": "custom-value"},
	})
	require.NoError(t, err)

	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.NoError(t, err)

	assert.Equal(t, "custom-value", capturedHeaders.Get("X-Custom-Auth"))
}

func TestOpenAI_GenerateChat_EndpointPathOverride(t *testing.T) {
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{
		HTTPClient:          srv.Client(),
		Secrets:             stubSecrets{byoa: "k"},
		Endpoint:            srv.URL,
		ChatCompletionsPath: "/custom/chat",
	})
	require.NoError(t, err)

	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.NoError(t, err)

	assert.Equal(t, "/custom/chat", capturedPath)
}

func TestOpenAI_GenerateChat_AbsoluteEndpointPathOverride(t *testing.T) {
	var capturedURL string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedURL = r.URL.String()
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{
		HTTPClient:          srv.Client(),
		Secrets:             stubSecrets{byoa: "k"},
		Endpoint:            srv.URL,
		ChatCompletionsPath: srv.URL + "/generate",
	})
	require.NoError(t, err)

	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.NoError(t, err)

	assert.Equal(t, "/generate", capturedURL)
}

func TestOpenAI_GenerateImage_B64JSON(t *testing.T) {
	pngBytes := []byte("\x89PNG\r\n\x1a\nfake-png-data")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(fmt.Sprintf(`{
			"data": [{"b64_json": "%s", "revised_prompt": "a cat"}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 1}
		}`, base64.StdEncoding.EncodeToString(pngBytes))))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.GenerateImage(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "dall-e-3",
		Prompt:         "a cat",
	})
	require.NoError(t, err)

	assert.Equal(t, pngBytes, resp.ImageBytes)
	assert.Equal(t, "image/png", resp.ImageMIMEType)
	assert.Equal(t, "a cat", resp.RevisedPrompt)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
}

func TestOpenAI_GenerateImage_URLFetch(t *testing.T) {
	// Serve the image bytes from a test server.
	imgSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/webp")
		_, _ = w.Write([]byte("RIFF\x24\x00\x00\x00WEBPVP8 "))
	}))
	defer imgSrv.Close()

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(fmt.Sprintf(`{
			"data": [{"url": "%s"}],
			"usage": {"prompt_tokens": 5, "completion_tokens": 1}
		}`, imgSrv.URL)))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.GenerateImage(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "dall-e-3",
		Prompt:         "a cat",
	})
	require.NoError(t, err)

	assert.Equal(t, "image/webp", resp.ImageMIMEType)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
}

func TestOpenAI_GenerateImage_NoData(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data": [], "usage": {}}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.GenerateImage(context.Background(), domain.VendorRequest{
		TenantID:       "tenant-1",
		LogicalModelID: "dall-e-3",
		Prompt:         "a cat",
	})
	require.NoError(t, err)

	assert.Equal(t, domain.FinishReasonUnspecified, resp.FinishReason)
	assert.Contains(t, resp.FinishDetail, "no image data")
}

func TestOpenAI_GenerateImage_RequiresPrompt(t *testing.T) {
	c, err := openai.New(openai.Config{Secrets: stubSecrets{byoa: "k"}})
	require.NoError(t, err)

	_, err = c.GenerateImage(context.Background(), domain.VendorRequest{LogicalModelID: "dall-e-3"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "prompt")
}

func TestOpenAI_EmbedText(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{
			"model": "text-embedding-3-small",
			"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
			"usage": {"prompt_tokens": 7}
		}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	resp, err := c.EmbedText(context.Background(), domain.EmbedVendorRequest{
		LogicalModelID:   "text-embedding-3-small",
		Text:             "hello",
		OutputDimensions: 256,
	})
	require.NoError(t, err)

	assert.Equal(t, "text-embedding-3-small", capturedBody["model"])
	assert.Equal(t, "hello", capturedBody["input"])
	assert.Equal(t, float64(256), capturedBody["dimensions"])

	assert.Equal(t, "text-embedding-3-small", resp.ModelVersion)
	assert.Equal(t, int64(7), resp.InputTokens)
	assert.Len(t, resp.Values, 3)
	assert.InDelta(t, 0.1, resp.Values[0], 0.001)
}

func TestOpenAI_EmbedText_NoDimensions(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{"model": "text-embedding-3-small", "data": [{"embedding": [0.1], "index": 0}], "usage": {"prompt_tokens": 1}}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	_, err = c.EmbedText(context.Background(), domain.EmbedVendorRequest{
		LogicalModelID: "text-embedding-3-small",
		Text:           "hello",
	})
	require.NoError(t, err)

	_, hasDimensions := capturedBody["dimensions"]
	assert.False(t, hasDimensions)
}

func TestOpenAI_EmbedText_EndpointPathOverride(t *testing.T) {
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		_, _ = w.Write([]byte(`{"model": "text-embedding-3-small", "data": [{"embedding": [0.1], "index": 0}], "usage": {"prompt_tokens": 1}}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{
		HTTPClient:     srv.Client(),
		Secrets:        stubSecrets{byoa: "k"},
		Endpoint:       srv.URL,
		EmbeddingsPath: "/custom/embeddings",
	})
	require.NoError(t, err)

	_, err = c.EmbedText(context.Background(), domain.EmbedVendorRequest{
		LogicalModelID: "text-embedding-3-small",
		Text:           "hello",
	})
	require.NoError(t, err)

	assert.Equal(t, "/custom/embeddings", capturedPath)
}

func TestOpenAI_ProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limit"}}`))
	}))
	defer srv.Close()

	c, err := openai.New(openai.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)

	_, err = c.Generate(context.Background(), domain.VendorRequest{LogicalModelID: "gpt-4o"})
	require.Error(t, err)

	var pe *openai.ProviderError
	if errors.As(err, &pe) {
		assert.Equal(t, 429, pe.Status)
		assert.Contains(t, pe.Endpoint, "/v1/chat/completions")
		assert.Contains(t, pe.Body, "rate limit")
	} else {
		t.Fatalf("expected ProviderError, got %T: %v", err, err)
	}
}

func TestOpenAI_SniffImageMime(t *testing.T) {
	tests := []struct {
		name     string
		data     []byte
		expected string
	}{
		{"png", []byte("\x89PNG\r\n\x1a\n"), "image/png"},
		{"jpeg", []byte{0xFF, 0xD8, 0xFF}, "image/jpeg"},
		{"webp", []byte("RIFF\x24\x00\x00\x00WEBPVP8 "), "image/webp"},
		{"gif87a", []byte("GIF87a"), "image/gif"},
		{"gif89a", []byte("GIF89a"), "image/gif"},
		{"unknown", []byte("unknown"), "image/png"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, openai.SniffImageMime(tt.data))
		})
	}
}
