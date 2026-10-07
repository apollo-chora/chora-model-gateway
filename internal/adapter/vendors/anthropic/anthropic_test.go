package anthropic_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/anthropic"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubSecrets struct {
	byoa    string
	byoaErr error
}

func (s stubSecrets) ResolveByoaKey(ctx context.Context, tenantID string, vendor domain.VendorFamily) (string, error) {
	return s.byoa, s.byoaErr
}
func (s stubSecrets) ResolveGlobalKey(ctx context.Context, vendor domain.VendorFamily) (string, error) {
	return "", nil
}

func TestAnthropic_Family(t *testing.T) {
	c, err := anthropic.New(anthropic.Config{Secrets: stubSecrets{}})
	require.NoError(t, err)
	assert.Equal(t, domain.VendorFamilyAnthropic, c.Family())
}

func TestAnthropic_GenerateHappyPath(t *testing.T) {
	var capturedKey, capturedVersion string
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedKey = r.Header.Get("x-api-key")
		capturedVersion = r.Header.Get("anthropic-version")
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{
			"model": "claude-opus-4-7-20251101",
			"stop_reason": "end_turn",
			"content": [{"type":"text","text":"two-week Sprint Planning is timeboxed at 4 hours"}],
			"usage": {"input_tokens": 100, "output_tokens": 30, "cache_read_input_tokens": 20}
		}`))
	}))
	defer srv.Close()

	c, err := anthropic.New(anthropic.Config{
		HTTPClient: srv.Client(),
		Secrets:    stubSecrets{byoa: "sk-ant-test"},
		Endpoint:   srv.URL,
	})
	require.NoError(t, err)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:         "tenant-1",
		LogicalModelID:   "claude-opus-4-7",
		Prompt:           "ping",
		SystemPrompt:     "you reply pong",
		GenerationConfig: map[string]any{"temperature": 0.3, "max_tokens": 256},
	})
	require.NoError(t, err)

	assert.Equal(t, "sk-ant-test", capturedKey)
	assert.Equal(t, "2023-06-01", capturedVersion)
	assert.Equal(t, "claude-opus-4-7", capturedBody["model"])
	assert.Equal(t, "you reply pong", capturedBody["system"])
	assert.Equal(t, 256.0, capturedBody["max_tokens"])
	assert.Equal(t, 0.3, capturedBody["temperature"])

	assert.Equal(t, "two-week Sprint Planning is timeboxed at 4 hours", resp.Completion)
	assert.Equal(t, "claude-opus-4-7-20251101", resp.ModelVersion)
	assert.Equal(t, int64(100), resp.Usage.InputTokens)
	assert.Equal(t, int64(30), resp.Usage.OutputTokens)
	assert.Equal(t, int64(20), resp.Usage.CachedTokens)
	// Pricing is micro-USD PER 1M tokens (claude-opus-4-7: 15_000_000 input +
	// 75_000_000 output + 1_500_000 cache-read + 18_750_000 cache-write).
	// Billable input = 100 - 0 cache writes - 20 cached = 80 tokens.
	// Cost = 80 * 15_000_000 / 1_000_000 + 20 * 1_500_000 / 1_000_000
	//      + 30 * 75_000_000 / 1_000_000
	//      = 1_200 + 30 + 2_250 = 3_480 micro-USD.
	assert.Equal(t, int64(3_480), resp.Usage.CostMicros)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
}

func TestAnthropic_GenerateNoBYOAKey(t *testing.T) {
	c, err := anthropic.New(anthropic.Config{Secrets: stubSecrets{}})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{TenantID: "tenant-1", LogicalModelID: "claude-opus-4-7"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no BYOA")
}

func TestAnthropic_GenerateBYOAError(t *testing.T) {
	c, err := anthropic.New(anthropic.Config{Secrets: stubSecrets{byoaErr: errors.New("SM denied")}})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{TenantID: "tenant-1", LogicalModelID: "claude-opus-4-7"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "BYOA")
}

func TestAnthropic_GenerateMaxTokens(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"stop_reason":"max_tokens","content":[{"type":"text","text":"truncated"}],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-opus-4-7"})
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonMaxTokens, resp.FinishReason)
}

func TestAnthropic_GenerateRefusal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"stop_reason":"refusal","content":[],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-opus-4-7"})
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
}

func TestAnthropic_GenerateHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"bad key"}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-opus-4-7"})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "401")
}

func TestAnthropic_NewValidation(t *testing.T) {
	_, err := anthropic.New(anthropic.Config{Secrets: nil})
	require.Error(t, err)
}

func TestAnthropic_GenerateUnknownModel_ZeroCost(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-vapor-X"})
	require.NoError(t, err)
	assert.Equal(t, int64(0), resp.Usage.CostMicros)
}

func TestAnthropic_GenerateCacheWrites(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"stop_reason": "end_turn",
			"content": [{"type":"text","text":"ok"}],
			"usage": {"input_tokens": 120, "output_tokens": 10, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 50}
		}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-opus-4-7"})
	require.NoError(t, err)
	// Plain input = 120 - 50 cache writes = 70. Billable = 70 - 20 cached = 50.
	// Cost = 50 * 15_000_000/1e6 + 20 * 1_500_000/1e6 + 50 * 18_750_000/1e6 + 10 * 75_000_000/1e6
	//      = 750 + 30 + 937 + 750 = 2_467 micro-USD (term-wise floored).
	assert.Equal(t, int64(120), resp.Usage.InputTokens)
	assert.Equal(t, int64(20), resp.Usage.CachedTokens)
	assert.Equal(t, int64(2_467), resp.Usage.CostMicros)
}

func TestAnthropic_GenerateGrounded(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{
			"stop_reason": "end_turn",
			"content": [
				{"type": "server_tool_use", "input": {"query": "most recent Formula 1 race winner"}},
				{"type": "text", "text": "Max Verstappen won."}
			],
			"usage": {}
		}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{
		HTTPClient: srv.Client(),
		Secrets:    stubSecrets{byoa: "k"},
		Endpoint:   srv.URL,
		Grounding:  &anthropic.GroundingConfig{MaxUses: 3},
	})
	require.NoError(t, err)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		TenantID:         "t",
		LogicalModelID:   "claude-sonnet-4-6",
		ResponseModality: "GROUNDED",
	})
	require.NoError(t, err)

	tools, ok := capturedBody["tools"].([]any)
	require.True(t, ok, "grounded request must carry tools, got %v", capturedBody["tools"])
	require.Len(t, tools, 1)
	tool := tools[0].(map[string]any)
	assert.Equal(t, "web_search_20250305", tool["type"])
	assert.Equal(t, "web_search", tool["name"])
	assert.Equal(t, float64(3), tool["max_uses"])

	assert.Equal(t, []string{"most recent Formula 1 race winner"}, resp.SearchQueries)
	assert.Equal(t, "Max Verstappen won.", resp.Completion)
}

func TestAnthropic_GenerateGroundedAppendsToCallerTools(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{
		HTTPClient: srv.Client(),
		Secrets:    stubSecrets{byoa: "k"},
		Endpoint:   srv.URL,
		Grounding:  &anthropic.GroundingConfig{ToolType: "web_search_20250305", ToolName: "search"},
	})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{
		TenantID:         "t",
		LogicalModelID:   "claude-sonnet-4-6",
		ResponseModality: "GROUNDED",
		ToolsJSON:        `[{"name":"get_weather","description":"weather","input_schema":{"type":"object"}}]`,
	})
	require.NoError(t, err)
	tools := capturedBody["tools"].([]any)
	require.Len(t, tools, 2)
	assert.Equal(t, "get_weather", tools[0].(map[string]any)["name"])
	assert.Equal(t, "search", tools[1].(map[string]any)["name"])
}

func TestAnthropic_GenerateUngroundedHasNoGroundingTool(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{
		HTTPClient: srv.Client(),
		Secrets:    stubSecrets{byoa: "k"},
		Endpoint:   srv.URL,
		Grounding:  &anthropic.GroundingConfig{},
	})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-sonnet-4-6"})
	require.NoError(t, err)
	_, present := capturedBody["tools"]
	assert.False(t, present, "ungrounded request must not carry tools")
}

func TestAnthropic_GenerationConfigBreadth(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "t",
		LogicalModelID: "claude-sonnet-4-6",
		GenerationConfig: map[string]any{
			"temperature":     0.7,
			"top_p":           0.9,
			"max_tokens":      float64(512),
			"n":               float64(2),
			"seed":            float64(42),
			"stop":            "END",
			"tool_choice":     "auto",
			"response_format": map[string]any{"type": "json_object"},
			// Bools are rejected by the Python runtime's _number coercion.
			"temperature_bool": true,
		},
	})
	require.NoError(t, err)
	assert.Equal(t, 0.7, capturedBody["temperature"])
	assert.Equal(t, 0.9, capturedBody["top_p"])
	assert.Equal(t, 512.0, capturedBody["max_tokens"])
	assert.Equal(t, 2.0, capturedBody["n"])
	assert.Equal(t, 42.0, capturedBody["seed"])
	assert.Equal(t, []any{"END"}, capturedBody["stop_sequences"])
	assert.Equal(t, "auto", capturedBody["tool_choice"])
	assert.Equal(t, map[string]any{"type": "json_object"}, capturedBody["response_format"])
	_, present := capturedBody["temperature_bool"]
	assert.False(t, present)
}

func TestAnthropic_GenerateStopSequencesList(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "t",
		LogicalModelID: "claude-sonnet-4-6",
		GenerationConfig: map[string]any{
			"stop": []any{"a", 7, "b"},
		},
	})
	require.NoError(t, err)
	// Non-string list elements are dropped, mirroring the Python runtime.
	assert.Equal(t, []any{"a", "b"}, capturedBody["stop_sequences"])
}

func TestAnthropic_GenerateCitations(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{
			"stop_reason": "end_turn",
			"content": [
				{"type": "text", "text": "Max won.", "citations": [
					{"url": "https://f1.example/race-report", "title": "Race report", "cited_text": "Max Verstappen won the race"},
					{"title": "no url — dropped"}
				]}
			],
			"usage": {}
		}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	resp, err := c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-sonnet-4-6"})
	require.NoError(t, err)
	require.Len(t, resp.Citations, 1)
	assert.Equal(t, domain.Citation{
		URL:     "https://f1.example/race-report",
		Title:   "Race report",
		Snippet: "Max Verstappen won the race",
	}, resp.Citations[0])
}

func TestAnthropic_GenerateContentsJSONMessages(t *testing.T) {
	var capturedBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&capturedBody)
		_, _ = w.Write([]byte(`{"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{
		TenantID:       "t",
		LogicalModelID: "claude-sonnet-4-6",
		ContentsJSON:   `[{"role":"user","content":"BEGIN"},{"role":"assistant","content":"prior"}]`,
	})
	require.NoError(t, err)
	assert.Equal(t, []any{
		map[string]any{"role": "user", "content": "BEGIN"},
		map[string]any{"role": "assistant", "content": "prior"},
	}, capturedBody["messages"])
}

func TestAnthropic_ExtraHeaders(t *testing.T) {
	var capturedHeaders http.Header
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedHeaders = r.Header.Clone()
		_, _ = w.Write([]byte(`{"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{
		HTTPClient:   srv.Client(),
		Secrets:      stubSecrets{byoa: "k"},
		Endpoint:     srv.URL,
		ExtraHeaders: map[string]string{"x-custom": "yes", "anthropic-beta": "prompt-caching-2024-07-31"},
	})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-sonnet-4-6"})
	require.NoError(t, err)
	assert.Equal(t, "yes", capturedHeaders.Get("x-custom"))
	assert.Equal(t, "prompt-caching-2024-07-31", capturedHeaders.Get("anthropic-beta"))
	// The standard headers still win over extra headers.
	assert.Equal(t, "k", capturedHeaders.Get("x-api-key"))
	assert.Equal(t, "2023-06-01", capturedHeaders.Get("anthropic-version"))
}

func TestAnthropic_MessagesPathOverride(t *testing.T) {
	var capturedPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedPath = r.URL.Path
		_, _ = w.Write([]byte(`{"stop_reason":"end_turn","content":[{"type":"text","text":"ok"}],"usage":{}}`))
	}))
	defer srv.Close()

	// Relative path is joined onto the endpoint.
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL, MessagesPath: "/proxy/messages"})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-sonnet-4-6"})
	require.NoError(t, err)
	assert.Equal(t, "/proxy/messages", capturedPath)

	// An absolute URL is used verbatim.
	c2, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: "https://ignored.example", MessagesPath: srv.URL + "/absolute"})
	require.NoError(t, err)
	_, err = c2.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-sonnet-4-6"})
	require.NoError(t, err)
	assert.Equal(t, "/absolute", capturedPath)
}

func TestAnthropic_ProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-sonnet-4-6"})
	require.Error(t, err)
	var perr *anthropic.ProviderError
	require.ErrorAs(t, err, &perr)
	assert.Equal(t, http.StatusTooManyRequests, perr.Status)
	assert.Equal(t, srv.URL+"/v1/messages", perr.Endpoint)
	assert.Contains(t, perr.Body, "rate_limit_error")
	assert.Equal(t, http.StatusTooManyRequests, perr.UpstreamStatus())
}

func TestAnthropic_ProviderErrorTruncatesBody(t *testing.T) {
	long := strings.Repeat("x", 5000)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(long))
	}))
	defer srv.Close()
	c, err := anthropic.New(anthropic.Config{HTTPClient: srv.Client(), Secrets: stubSecrets{byoa: "k"}, Endpoint: srv.URL})
	require.NoError(t, err)
	_, err = c.Generate(context.Background(), domain.VendorRequest{TenantID: "t", LogicalModelID: "claude-sonnet-4-6"})
	require.Error(t, err)
	var perr *anthropic.ProviderError
	require.ErrorAs(t, err, &perr)
	assert.Equal(t, 2049, len([]rune(perr.Body))) // 2048 chars + ellipsis
	assert.True(t, strings.HasSuffix(perr.Body, "…"))
}
