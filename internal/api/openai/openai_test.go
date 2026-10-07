package openai_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/api/openai"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/apollo-chora/chora-model-gateway/internal/registry"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeInvoker struct {
	resp domain.InvokeResponse
	err  error
}

func (f *fakeInvoker) Execute(ctx context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error) {
	return f.resp, f.err
}

type fakeEmbedder struct {
	resp domain.EmbedFlowResponse
	err  error
}

func (f *fakeEmbedder) Embed(ctx context.Context, req domain.EmbedFlowRequest) (domain.EmbedFlowResponse, error) {
	return f.resp, f.err
}

type fakePinger struct{ err error }

func (f *fakePinger) Ping(ctx context.Context) error { return f.err }

type fakeRegistry struct{ specs []registry.ModelSpec }

func (r *fakeRegistry) Get(id string) (registry.ModelSpec, bool) {
	for _, s := range r.specs {
		if strings.EqualFold(s.ID, id) {
			return s, true
		}
	}
	return registry.ModelSpec{}, false
}

func (r *fakeRegistry) List() []registry.ModelSpec {
	out := make([]registry.ModelSpec, len(r.specs))
	copy(out, r.specs)
	return out
}

func (r *fakeRegistry) Resolve(id string, callerFallbacks []string) (registry.ModelSpec, []registry.ModelSpec, error) {
	primary, ok := r.Get(id)
	if !ok {
		return registry.ModelSpec{}, nil, errors.New("not found")
	}
	return primary, nil, nil
}

func testRegistry() registry.Registry {
	return &fakeRegistry{specs: []registry.ModelSpec{
		{ID: "chatty", Provider: "openai", BaseURL: "https://api.example.test/v1", Capabilities: []string{"chat", "tools"}, ContextWindow: 8000, MaxOutputTokens: 512},
		{ID: "painter", Provider: "openai", BaseURL: "https://api.example.test/v1", Capabilities: []string{"image"}},
		{ID: "embedder", Provider: "openai", BaseURL: "https://api.example.test/v1", Capabilities: []string{"embeddings"}},
		{ID: "searcher", Provider: "openai", BaseURL: "https://api.example.test/v1", Capabilities: []string{"chat", "web_search"}, Grounding: &registry.GroundingSpec{Surface: "responses"}},
		{ID: "chatsearcher", Provider: "openai", BaseURL: "https://api.example.test/v1", Capabilities: []string{"chat", "web_search"}, Grounding: &registry.GroundingSpec{Surface: "chat_completions"}},
	}}
}

func newTestServer(t *testing.T, invoker openai.Invoker, embedder openai.Embedder) *openai.Server {
	t.Helper()
	srv, err := openai.NewServer(invoker, embedder, testRegistry(), &fakePinger{}, openai.Settings{
		DefaultTenantID: "tenant-1", DefaultGCID: "gcid-1", DefaultAgentID: "openai_compat",
	})
	require.NoError(t, err)
	return srv
}

func doRequest(t *testing.T, handler http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	return w
}

func TestHealthz(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "GET", "/healthz", "")
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "ok", w.Body.String())
	assert.Equal(t, "nosniff", w.Header().Get("X-Content-Type-Options"))
}

func TestReadyz_Ready(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "GET", "/readyz", "")
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "ready", w.Body.String())
}

func TestReadyz_NotReady(t *testing.T) {
	srv, err := openai.NewServer(&fakeInvoker{}, &fakeEmbedder{}, testRegistry(), &fakePinger{err: errors.New("db down")}, openai.Settings{})
	require.NoError(t, err)
	w := doRequest(t, srv.Handler(), "GET", "/readyz", "")
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Equal(t, "not ready", w.Body.String())
}

func TestListModels(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "GET", "/v1/models", "")
	assert.Equal(t, http.StatusOK, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "list", data["object"])
	assert.Len(t, data["data"], 5)
}

func TestGetModel(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "GET", "/v1/models/chatty", "")
	assert.Equal(t, http.StatusOK, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "chatty", data["id"])
	assert.Equal(t, "openai", data["owned_by"])
}

func TestGetModel_NotFound(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "GET", "/v1/models/ghost", "")
	assert.Equal(t, http.StatusNotFound, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "model_not_found", data["error"].(map[string]any)["code"])
}

func TestChat_HappyPath(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", Completion: "hello", Vendor: "openai",
		Usage: domain.TokenUsage{InputTokens: 3, OutputTokens: 2},
		ModelVersion: "chatty-upstream", FinishReason: domain.FinishReasonComplete,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusOK, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "chat.completion", data["object"])
	choices := data["choices"].([]any)
	msg := choices[0].(map[string]any)["message"].(map[string]any)
	assert.Equal(t, "hello", msg["content"])
	assert.Equal(t, "stop", choices[0].(map[string]any)["finish_reason"])
	assert.Equal(t, float64(5), data["usage"].(map[string]any)["total_tokens"])
}

func TestChat_StreamingRejected(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	assert.Equal(t, http.StatusNotImplemented, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "streaming_unsupported", data["error"].(map[string]any)["code"])
}

func TestChat_MissingModel(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "missing_parameter", data["error"].(map[string]any)["code"])
}

func TestChat_UnknownModel(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"ghost","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusNotFound, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "model_not_found", data["error"].(map[string]any)["code"])
}

func TestChat_MalformedJSON(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions", `{"model":`)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "invalid_json", data["error"].(map[string]any)["code"])
}

func TestChat_EmptyConversation(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"system","content":"be terse"}]}`)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "empty_conversation", data["error"].(map[string]any)["code"])
}

func TestChat_UpstreamStatusRelayed(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.InvokeError{
		Reason: domain.FinishReasonVendorError, Detail: "upstream failed",
		UpstreamStatus: intPtr(429),
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusTooManyRequests, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	errObj := data["error"].(map[string]any)
	assert.Equal(t, "rate_limit_error", errObj["type"])
	assert.Equal(t, "upstream_error", errObj["code"])
}

func TestChat_BudgetBlock(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", FinishReason: domain.FinishReasonBudgetBlock,
		FinishDetail: "tenant LLM budget exhausted; policy=block",
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusPaymentRequired, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	errObj := data["error"].(map[string]any)
	assert.Equal(t, "insufficient_quota", errObj["type"])
	assert.Equal(t, "budget_exhausted", errObj["code"])
}

func TestChat_ManaBlock(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", FinishReason: domain.FinishReasonManaBlock,
		FinishDetail: "insufficient_mana required=100 available=50",
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusPaymentRequired, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	errObj := data["error"].(map[string]any)
	assert.Equal(t, "insufficient_quota", errObj["type"])
	assert.Equal(t, "insufficient_mana", errObj["code"])
}

func TestChat_ArmorBlock(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", FinishReason: domain.FinishReasonModelArmorBlock,
		FinishDetail: "model armor PRE blocked the prompt",
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusForbidden, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "model_armor_block", data["error"].(map[string]any)["code"])
}

func TestChat_PreconditionError_SurfaceUnstamped(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.PreconditionError{Reason: domain.DenySurfaceUnstamped, Detail: "surface required"}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "surface_unstamped", data["error"].(map[string]any)["code"])
}

func TestChat_PreconditionError_CompanionSuspended(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.PreconditionError{Reason: domain.DenyCompanionSuspended, Detail: "companion contained"}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusForbidden, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "companion_suspended", data["error"].(map[string]any)["code"])
}

func TestChat_IdentityFromHeaders(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", Completion: "ok", Vendor: "openai",
		Usage: domain.TokenUsage{InputTokens: 1, OutputTokens: 1}, FinishReason: domain.FinishReasonComplete,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-chora-tenant-id", "tenant-42")
	req.Header.Set("x-chora-gcid", "gcid-99")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestChat_IdentityFromQuery(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", Completion: "ok", Vendor: "openai",
		Usage: domain.TokenUsage{InputTokens: 1, OutputTokens: 1}, FinishReason: domain.FinishReasonComplete,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions?tenant=tenant-77",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestChat_TraceparentForwarded(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", Completion: "ok", Vendor: "openai",
		Usage: domain.TokenUsage{InputTokens: 1, OutputTokens: 1}, FinishReason: domain.FinishReasonComplete,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	req := httptest.NewRequest("POST", "/v1/chat/completions",
		strings.NewReader(`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("traceparent", "00-abc-def-01")
	req.Header.Set("tracestate", "foo=bar")
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestChat_GroundingHonoured(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", Completion: "grounded answer", Vendor: "openai",
		Usage: domain.TokenUsage{InputTokens: 10, OutputTokens: 5},
		FinishReason: domain.FinishReasonComplete, SearchQueries: []string{"test query"},
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatsearcher","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search"}]}`)
	assert.Equal(t, http.StatusOK, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	meta := data["chora_gateway"].(map[string]any)
	assert.Equal(t, true, meta["grounded"])
}

func TestChat_GroundingWrongSurface(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"searcher","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search"}]}`)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "wrong_grounding_surface", data["error"].(map[string]any)["code"])
}

func TestChat_GroundingNonGroundedModel(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"web_search"}]}`)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "grounding_unavailable", data["error"].(map[string]any)["code"])
}

func TestResponses_HappyPath(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", Completion: "answer", Vendor: "openai",
		Usage: domain.TokenUsage{InputTokens: 5, OutputTokens: 3}, FinishReason: domain.FinishReasonComplete,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/responses", `{"model":"chatty","input":"hi"}`)
	assert.Equal(t, http.StatusOK, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "response", data["object"])
	assert.Len(t, data["output"], 1)
}

func TestResponses_Grounded(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", Completion: "grounded", Vendor: "openai",
		Usage: domain.TokenUsage{InputTokens: 10, OutputTokens: 5},
		FinishReason: domain.FinishReasonComplete, SearchQueries: []string{"q1"},
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/responses",
		`{"model":"searcher","input":"hi","tools":[{"type":"web_search"}]}`)
	assert.Equal(t, http.StatusOK, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Len(t, data["output"], 2)
	assert.Equal(t, "web_search_call", data["output"].([]any)[0].(map[string]any)["type"])
	meta := data["chora_gateway"].(map[string]any)
	assert.Equal(t, true, meta["grounded"])
}

func TestResponses_StreamingRejected(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/responses",
		`{"model":"chatty","input":"hi","stream":true}`)
	assert.Equal(t, http.StatusNotImplemented, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "streaming_unsupported", data["error"].(map[string]any)["code"])
}

func TestResponses_ConversationInput(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", Completion: "ok", Vendor: "openai",
		Usage: domain.TokenUsage{InputTokens: 1, OutputTokens: 1}, FinishReason: domain.FinishReasonComplete,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/responses",
		`{"model":"chatty","input":[{"role":"user","content":"one"},{"role":"assistant","content":"two"}]}`)
	assert.Equal(t, http.StatusOK, w.Code)
}

func TestImages_HappyPath(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{
		InvocationID: "inv-1", ImageBytes: []byte{0x89, 0x50, 0x4E, 0x47},
		ImageMIMEType: "image/png", Vendor: "openai",
		Usage: domain.TokenUsage{InputTokens: 5, OutputTokens: 1}, FinishReason: domain.FinishReasonComplete,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/images/generations",
		`{"model":"painter","prompt":"a cat"}`)
	assert.Equal(t, http.StatusOK, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "image/png", data["mime_type"])
}

func TestImages_URLFormatRejected(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/images/generations",
		`{"model":"painter","prompt":"cat","response_format":"url"}`)
	assert.Equal(t, http.StatusNotImplemented, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "url_response_unsupported", data["error"].(map[string]any)["code"])
}

func TestImages_NoImageReturned(t *testing.T) {
	invoker := &fakeInvoker{resp: domain.InvokeResponse{InvocationID: "inv-1", FinishReason: domain.FinishReasonComplete}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/images/generations",
		`{"model":"painter","prompt":"cat"}`)
	assert.Equal(t, http.StatusBadGateway, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "no_image", data["error"].(map[string]any)["code"])
}

func TestEmbeddings_HappyPath(t *testing.T) {
	embedder := &fakeEmbedder{resp: domain.EmbedFlowResponse{
		InvocationID: "inv-1", Values: []float32{0.5, 0.25},
		Usage: domain.TokenUsage{InputTokens: 4},
	}}
	srv := newTestServer(t, &fakeInvoker{}, embedder)
	w := doRequest(t, srv.Handler(), "POST", "/v1/embeddings",
		`{"model":"embedder","input":"hello"}`)
	assert.Equal(t, http.StatusOK, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "list", data["object"])
	assert.Len(t, data["data"], 1)
	assert.Equal(t, float64(4), data["usage"].(map[string]any)["total_tokens"])
}

func TestEmbeddings_Batch(t *testing.T) {
	embedder := &fakeEmbedder{resp: domain.EmbedFlowResponse{
		InvocationID: "inv-1", Values: []float32{0.5},
		Usage: domain.TokenUsage{InputTokens: 2},
	}}
	srv := newTestServer(t, &fakeInvoker{}, embedder)
	w := doRequest(t, srv.Handler(), "POST", "/v1/embeddings",
		`{"model":"embedder","input":["a","b","c"]}`)
	assert.Equal(t, http.StatusOK, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Len(t, data["data"], 3)
}

func TestEmbeddings_Base64Rejected(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/embeddings",
		`{"model":"embedder","input":"hi","encoding_format":"base64"}`)
	assert.Equal(t, http.StatusNotImplemented, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "base64_unsupported", data["error"].(map[string]any)["code"])
}

func TestEmbeddings_UnknownModel(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/embeddings",
		`{"model":"ghost","input":"hi"}`)
	assert.Equal(t, http.StatusNotFound, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "model_not_found", data["error"].(map[string]any)["code"])
}

func TestMethodGuard_ChatGet(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "GET", "/v1/chat/completions", "")
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	errObj := data["error"].(map[string]any)
	assert.Equal(t, "method_not_allowed", errObj["code"])
	assert.Equal(t, "POST only", errObj["message"])
}

func TestMethodGuard_ModelsPost(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/models", `{}`)
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	errObj := data["error"].(map[string]any)
	assert.Equal(t, "method_not_allowed", errObj["code"])
	assert.Equal(t, "GET only", errObj["message"])
}

func TestMethodGuard_HealthzPost(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/healthz", `{}`)
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
}

func TestMethodGuard_ResponsesGet(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "GET", "/v1/responses", "")
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
}

func TestMethodGuard_EmbeddingsPut(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "PUT", "/v1/embeddings", `{}`)
	assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
}

func TestErrorEnvelope_Shape(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"ghost","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusNotFound, w.Code)
	assert.Equal(t, "application/json", w.Header().Get("Content-Type"))
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	errObj := data["error"].(map[string]any)
	assert.NotEmpty(t, errObj["message"])
	assert.NotEmpty(t, errObj["type"])
	assert.NotEmpty(t, errObj["code"])
}

func TestNosniffHeader_OnEveryResponse(t *testing.T) {
	srv := newTestServer(t, &fakeInvoker{}, &fakeEmbedder{})
	paths := []string{"/healthz", "/v1/models", "/v1/chat/completions"}
	for _, path := range paths {
		w := doRequest(t, srv.Handler(), "GET", path, "")
		assert.Equal(t, "nosniff", w.Header().Get("X-Content-Type-Options"), "path: %s", path)
	}
}

func TestUpstreamErrorType_401(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.InvokeError{
		Reason: domain.FinishReasonVendorError, Detail: "auth", UpstreamStatus: intPtr(401),
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusUnauthorized, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "authentication_error", data["error"].(map[string]any)["type"])
}

func TestUpstreamErrorType_403(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.InvokeError{
		Reason: domain.FinishReasonVendorError, Detail: "forbidden", UpstreamStatus: intPtr(403),
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusForbidden, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "authentication_error", data["error"].(map[string]any)["type"])
}

func TestUpstreamErrorType_404(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.InvokeError{
		Reason: domain.FinishReasonVendorError, Detail: "not found", UpstreamStatus: intPtr(404),
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusNotFound, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "invalid_request_error", data["error"].(map[string]any)["type"])
}

func TestUpstreamErrorType_502(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.InvokeError{
		Reason: domain.FinishReasonVendorError, Detail: "bad gateway", UpstreamStatus: intPtr(502),
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusBadGateway, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "upstream_error", data["error"].(map[string]any)["type"])
}

func TestInvokeError_ConfigError(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.InvokeError{
		Reason: domain.FinishReasonVendorError, Detail: "config error", Kind: domain.ErrorKindConfig,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusInternalServerError, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "server_error", data["error"].(map[string]any)["type"])
}

func TestInvokeError_BareError(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.InvokeError{
		Reason: domain.FinishReasonVendorError, Detail: "bad request", Kind: domain.ErrorKindBare,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "invalid_request_error", data["error"].(map[string]any)["type"])
}

func TestInvokeError_InvokeError(t *testing.T) {
	invoker := &fakeInvoker{err: &domain.InvokeError{
		Reason: domain.FinishReasonVendorError, Detail: "vendor error", Kind: domain.ErrorKindInvoke,
	}}
	srv := newTestServer(t, invoker, &fakeEmbedder{})
	w := doRequest(t, srv.Handler(), "POST", "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	assert.Equal(t, http.StatusBadGateway, w.Code)
	var data map[string]any
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &data))
	assert.Equal(t, "upstream_error", data["error"].(map[string]any)["type"])
}

func intPtr(i int) *int { return &i }
