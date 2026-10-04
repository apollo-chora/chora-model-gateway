package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/config"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// ---------------------------------------------------------------------------
// Doubles
// ---------------------------------------------------------------------------

type stubGateway struct {
	invoke func(ctx context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error)
	embed  func(ctx context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error)

	lastInvoke domain.InvokeRequest
	invokeN    int
}

func (s *stubGateway) Invoke(ctx context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error) {
	s.lastInvoke = req
	s.invokeN++
	if s.invoke != nil {
		return s.invoke(ctx, req)
	}
	return domain.InvokeResponse{
		InvocationID: "inv-1",
		Completion:   "hello from the stub",
		ModelVersion: "upstream-1",
		Vendor:       "openai",
		Usage:        domain.TokenUsage{InputTokens: 3, OutputTokens: 2},
		CompletedAt:  time.Unix(1_700_000_000, 0),
		FinishReason: domain.FinishReasonComplete,
	}, nil
}

func (s *stubGateway) Embed(ctx context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error) {
	if s.embed != nil {
		return s.embed(ctx, req)
	}
	return domain.EmbedResponse{
		InvocationID: "emb-1",
		Values:       []float32{0.5, 0.25},
		Usage:        domain.TokenUsage{InputTokens: 4},
		CompletedAt:  time.Unix(1_700_000_000, 0),
	}, nil
}

// testRegistry builds a registry covering the cases the facade must handle:
// an OpenAI chat model, an image model, an embeddings model, and a
// chat-only model (so the capability gate has something to refuse).
func testRegistry(t *testing.T) *config.Registry {
	t.Helper()
	reg, err := config.LoadRegistry(writeTempRegistry(t, `
models:
  - id: chatty
    provider: openai
    base_url: https://api.example.test/v1
    context_window: 8000
    max_output_tokens: 512
    capabilities: [chat, tools, vision]
    pricing:
      input_per_mtok_usd_micros: 1000000
      output_per_mtok_usd_micros: 2000000

  - id: painter
    provider: openai
    base_url: https://api.example.test/v1
    capabilities: [image]

  - id: embedder
    provider: openai
    base_url: https://api.example.test/v1
    capabilities: [embeddings]

  - id: textonly
    provider: openai
    base_url: https://api.example.test/v1
    capabilities: [chat]
`))
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	return reg
}

func newTestServer(t *testing.T, gw Gateway) *Server {
	t.Helper()
	s, err := NewServer(Config{
		Gateway:         gw,
		Models:          testRegistry(t),
		Version:         "test",
		DefaultTenantID: "00000000-0000-7000-8000-000000000001",
		DefaultGCID:     "00000000-0000-7000-8000-000000000002",
		DefaultAgentID:  "openai_compat",
	})
	if err != nil {
		t.Fatalf("NewServer: %v", err)
	}
	return s
}

func do(t *testing.T, s *Server, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	var reader *strings.Reader
	if body == "" {
		reader = strings.NewReader("")
	} else {
		reader = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, path, reader)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	s.ServeHTTP(rec, req)
	return rec
}

func decodeError(t *testing.T, rec *httptest.ResponseRecorder) apiError {
	t.Helper()
	var e apiError
	if err := json.Unmarshal(rec.Body.Bytes(), &e); err != nil {
		t.Fatalf("error body is not the OpenAI error envelope: %v (%s)", err, rec.Body.String())
	}
	return e
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

func TestNewServer_RequiredFields(t *testing.T) {
	reg := testRegistry(t)
	cases := []struct {
		name string
		cfg  Config
	}{
		{"no gateway", Config{Models: reg, DefaultTenantID: "t"}},
		{"no registry", Config{Gateway: &stubGateway{}, DefaultTenantID: "t"}},
		// The budget row is tenant-scoped, so there is no sensible default
		// for the tenant; refusing beats attributing spend to "".
		{"no tenant", Config{Gateway: &stubGateway{}, Models: reg}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := NewServer(tc.cfg); err == nil {
				t.Fatal("expected an error")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

func TestHandleModels_ListsTheRegistry(t *testing.T) {
	rec := do(t, newTestServer(t, &stubGateway{}), http.MethodGet, "/v1/models", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var out modelList
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Object != "list" {
		t.Errorf("object = %q, want list", out.Object)
	}
	if len(out.Data) != 4 {
		t.Errorf("listed %d models, want 4", len(out.Data))
	}
	// The additive fields are how an operator's tooling learns the limits
	// without a second call.
	var chatty *modelEntry
	for i := range out.Data {
		if out.Data[i].ID == "chatty" {
			chatty = &out.Data[i]
		}
	}
	if chatty == nil {
		t.Fatal("chatty missing from the list")
	}
	if chatty.ContextWindow != 8000 || chatty.MaxOutputTokens != 512 {
		t.Errorf("limits = %d/%d, want 8000/512", chatty.ContextWindow, chatty.MaxOutputTokens)
	}
}

func TestHandleModels_SingleRetrieval(t *testing.T) {
	s := newTestServer(t, &stubGateway{})
	rec := do(t, s, http.MethodGet, "/v1/models/chatty", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	var out modelEntry
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.ID != "chatty" {
		t.Errorf("id = %q", out.ID)
	}

	rec = do(t, s, http.MethodGet, "/v1/models/ghost", "")
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404 for an unknown model", rec.Code)
	}
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

func TestChatCompletions_HappyPath(t *testing.T) {
	gw := &stubGateway{}
	rec := do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	var out chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Object != "chat.completion" {
		t.Errorf("object = %q", out.Object)
	}
	if len(out.Choices) != 1 || out.Choices[0].FinishReason != "stop" {
		t.Fatalf("choices = %+v", out.Choices)
	}
	if out.Choices[0].Message.Role != "assistant" {
		t.Errorf("role = %q, want assistant", out.Choices[0].Message.Role)
	}
	if out.Usage.TotalTokens != 5 {
		t.Errorf("total tokens = %d, want 3+2", out.Usage.TotalTokens)
	}
	// The additive block tells an operator what the gateway actually did.
	if out.Gateway == nil || out.Gateway.Vendor != "openai" {
		t.Errorf("gateway meta = %+v", out.Gateway)
	}
}

func TestChatCompletions_StripsSystemPrompt(t *testing.T) {
	gw := &stubGateway{}
	do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[
			{"role":"system","content":"be terse"},
			{"role":"developer","content":"and precise"},
			{"role":"user","content":"hi"}]}`)

	if gw.lastInvoke.SystemPrompt != "be terse\n\nand precise" {
		t.Errorf("system prompt = %q, want both instructions joined", gw.lastInvoke.SystemPrompt)
	}
	// The conversation that reaches the vendor must not carry the system turn.
	if strings.Contains(gw.lastInvoke.ContentsJSON, "be terse") {
		t.Errorf("contents leaked the system prompt: %s", gw.lastInvoke.ContentsJSON)
	}
	if !strings.Contains(gw.lastInvoke.ContentsJSON, `"role":"user"`) {
		t.Errorf("contents lost the user turn: %s", gw.lastInvoke.ContentsJSON)
	}
}

func TestChatCompletions_ForwardsGenerationConfig(t *testing.T) {
	gw := &stubGateway{}
	do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}],
		  "temperature":0.2,"top_p":0.9,"max_tokens":123,"seed":7,"stop":["END"]}`)

	cfg := gw.lastInvoke.GenerationConfig
	if cfg["temperature"] != 0.2 {
		t.Errorf("temperature = %v", cfg["temperature"])
	}
	if cfg["top_p"] != 0.9 {
		t.Errorf("top_p = %v", cfg["top_p"])
	}
	if cfg["max_tokens"] != float64(123) {
		t.Errorf("max_tokens = %v (%T), want float64(123)", cfg["max_tokens"], cfg["max_tokens"])
	}
	if cfg["seed"] != float64(7) {
		t.Errorf("seed = %v (%T), want float64(7) so the map's numeric types agree", cfg["seed"], cfg["seed"])
	}
	// An array stop sequence stays an array; the vendor adapters accept both
	// the string and the slice form.
	stop, ok := cfg["stop"].([]string)
	if !ok || len(stop) != 1 || stop[0] != "END" {
		t.Errorf("stop = %#v, want []string{END}", cfg["stop"])
	}
}

func TestChatCompletions_StopStringForm(t *testing.T) {
	gw := &stubGateway{}
	do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}],"stop":"END"}`)
	if gw.lastInvoke.GenerationConfig["stop"] != "END" {
		t.Errorf("stop = %#v, want the bare string preserved", gw.lastInvoke.GenerationConfig["stop"])
	}
}

func TestChatCompletions_PrefersMaxCompletionTokens(t *testing.T) {
	gw := &stubGateway{}
	do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}],"max_completion_tokens":55}`)
	if gw.lastInvoke.GenerationConfig["max_tokens"] != float64(55) {
		t.Errorf("max_tokens = %v, want max_completion_tokens honoured", gw.lastInvoke.GenerationConfig["max_tokens"])
	}
}

func TestChatCompletions_ForwardsTools(t *testing.T) {
	gw := &stubGateway{}
	do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}],
		  "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]}`)
	if !strings.Contains(gw.lastInvoke.ToolsJSON, `"lookup"`) {
		t.Errorf("tools lost: %s", gw.lastInvoke.ToolsJSON)
	}
}

func TestChatCompletions_Attribution(t *testing.T) {
	gw := &stubGateway{}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"chatty","messages":[{"role":"user","content":"hi"}],"user":"alice"}`))
	req.Header.Set("X-Chora-Tenant-Id", "tenant-42")
	req.Header.Set("X-Chora-Gcid", "gcid-99")
	rec := httptest.NewRecorder()
	newTestServer(t, gw).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d", rec.Code)
	}
	if gw.lastInvoke.TenantID != "tenant-42" {
		t.Errorf("tenant = %q, want the header value", gw.lastInvoke.TenantID)
	}
	if gw.lastInvoke.GCID != "gcid-99" {
		t.Errorf("gcid = %q", gw.lastInvoke.GCID)
	}
	// The `user` field is the OpenAI-native attribution hook; it becomes the
	// ledger row's role tag.
	if gw.lastInvoke.ActionCode != "alice" {
		t.Errorf("action code = %q, want the caller-supplied user", gw.lastInvoke.ActionCode)
	}
}

func TestChatCompletions_UnknownModelIs404(t *testing.T) {
	rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/chat/completions",
		`{"model":"ghost","messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404", rec.Code)
	}
	if code := decodeError(t, rec).Error.Code; code != "model_not_found" {
		t.Errorf("code = %q", code)
	}
}

func TestChatCompletions_StreamingIsRefusedExplicitly(t *testing.T) {
	gw := &stubGateway{}
	rec := do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}],"stream":true}`)
	// A client that asked for SSE and silently receives a JSON blob hangs or
	// mis-parses. An explicit 501 is the honest answer.
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501", rec.Code)
	}
	if gw.invokeN != 0 {
		t.Error("a refused streaming request still reached the gateway")
	}
}

func TestChatCompletions_SystemOnlyIsRejected(t *testing.T) {
	gw := &stubGateway{}
	rec := do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"system","content":"be terse"}]}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	if gw.invokeN != 0 {
		t.Error("an empty conversation still reached the gateway")
	}
}

func TestChatCompletions_MissingFields(t *testing.T) {
	cases := []struct{ name, body string }{
		{"no model", `{"messages":[{"role":"user","content":"hi"}]}`},
		{"no messages", `{"model":"chatty"}`},
		{"empty messages", `{"model":"chatty","messages":[]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/chat/completions", tc.body)
			if rec.Code != http.StatusBadRequest {
				t.Errorf("status = %d, want 400", rec.Code)
			}
		})
	}
}

func TestChatCompletions_MalformedJSON(t *testing.T) {
	rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/chat/completions", `{"model":`)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestChatCompletions_WrongMethod(t *testing.T) {
	rec := do(t, newTestServer(t, &stubGateway{}), http.MethodGet, "/v1/chat/completions", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestChatCompletions_BudgetBlockIs402(t *testing.T) {
	gw := &stubGateway{invoke: func(context.Context, domain.InvokeRequest) (domain.InvokeResponse, error) {
		return domain.InvokeResponse{
			InvocationID: "inv-1",
			FinishReason: domain.FinishReasonBudgetBlock,
			FinishDetail: "tenant LLM budget exhausted; policy=block",
		}, nil
	}}
	rec := do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)

	// Returning 200 with an empty completion would tell the caller the model
	// answered and said nothing.
	if rec.Code != http.StatusPaymentRequired {
		t.Fatalf("status = %d, want 402", rec.Code)
	}
	e := decodeError(t, rec)
	if e.Error.Code != "budget_exhausted" {
		t.Errorf("code = %q", e.Error.Code)
	}
}

func TestChatCompletions_ConfigErrorIs500(t *testing.T) {
	gw := &stubGateway{invoke: func(context.Context, domain.InvokeRequest) (domain.InvokeResponse, error) {
		return domain.InvokeResponse{}, &domain.ConfigError{Detail: `credential "X" is set but empty`}
	}}
	rec := do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)

	// 500, not 502: the provider is fine, the gateway is misconfigured.
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	if code := decodeError(t, rec).Error.Code; code != "gateway_misconfigured" {
		t.Errorf("code = %q", code)
	}
}

func TestChatCompletions_UpstreamStatusIsRelayed(t *testing.T) {
	cases := []struct {
		name       string
		status     int
		wantStatus int
	}{
		{"unauthorized", http.StatusUnauthorized, http.StatusUnauthorized},
		{"rate limited", http.StatusTooManyRequests, http.StatusTooManyRequests},
		{"not found", http.StatusNotFound, http.StatusNotFound},
		{"server error", http.StatusBadGateway, http.StatusBadGateway},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gw := &stubGateway{invoke: func(context.Context, domain.InvokeRequest) (domain.InvokeResponse, error) {
				return domain.InvokeResponse{}, &domain.InvokeError{
					Reason: domain.FinishReasonVendorError,
					Detail: "upstream failed",
					Inner:  &upstreamStubError{status: tc.status},
				}
			}}
			rec := do(t, newTestServer(t, gw), http.MethodPost, "/v1/chat/completions",
				`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
			// A provider's own status is more useful to a caller than a
			// flattened 502: a 401 means the key is wrong.
			if rec.Code != tc.wantStatus {
				t.Errorf("status = %d, want %d", rec.Code, tc.wantStatus)
			}
		})
	}
}

type upstreamStubError struct{ status int }

func (e *upstreamStubError) Error() string           { return "upstream" }
func (e *upstreamStubError) UpstreamStatusCode() int { return e.status }

// ---------------------------------------------------------------------------
// POST /v1/images/generations
// ---------------------------------------------------------------------------

func TestImageGenerations_HappyPath(t *testing.T) {
	gw := &stubGateway{invoke: func(_ context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error) {
		if req.ResponseModality != domain.ModalityImage {
			t.Errorf("modality = %q, want IMAGE", req.ResponseModality)
		}
		return domain.InvokeResponse{
			InvocationID:       "inv-1",
			ImageBytes:         []byte("\x89PNG-bytes"),
			ImageMIMEType:      "image/png",
			ImageRevisedPrompt: "a sharper cat",
			Usage:              domain.TokenUsage{InputTokens: 5},
			FinishReason:       domain.FinishReasonComplete,
			CompletedAt:        time.Unix(1_700_000_000, 0),
		}, nil
	}}
	rec := do(t, newTestServer(t, gw), http.MethodPost, "/v1/images/generations",
		`{"model":"painter","prompt":"a cat","size":"1024x1024","quality":"high"}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}
	var out imageResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(out.Data) != 1 {
		t.Fatalf("data = %d entries, want 1", len(out.Data))
	}
	if out.Data[0].B64JSON == "" {
		t.Error("b64_json is empty")
	}
	if out.Data[0].RevisedPrompt != "a sharper cat" {
		t.Errorf("revised prompt = %q", out.Data[0].RevisedPrompt)
	}
	// The size/quality knobs are forwarded to the images surface only.
	if gw.lastInvoke.GenerationConfig["size"] != "1024x1024" {
		t.Errorf("size = %v", gw.lastInvoke.GenerationConfig["size"])
	}
	if gw.lastInvoke.GenerationConfig["quality"] != "high" {
		t.Errorf("quality = %v", gw.lastInvoke.GenerationConfig["quality"])
	}
}

func TestImageGenerations_URLFormatIsRefused(t *testing.T) {
	// The gateway returns bytes and hosts nothing, so a URL would point at
	// nothing. Saying so beats returning a dead link — and it must be refused
	// before the dispatch, not after, or a provider image generation gets paid
	// for and then discarded.
	gw := &stubGateway{}
	rec := do(t, newTestServer(t, gw), http.MethodPost, "/v1/images/generations",
		`{"model":"painter","prompt":"cat","response_format":"url"}`)
	if gw.invokeN != 0 {
		t.Error("an unsupported response_format still spent a provider call")
	}
	if rec.Code != http.StatusNotImplemented {
		t.Errorf("status = %d, want 501", rec.Code)
	}
	if code := decodeError(t, rec).Error.Code; code != "url_response_unsupported" {
		t.Errorf("code = %q", code)
	}
}

func TestImageGenerations_RequiresPrompt(t *testing.T) {
	rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/images/generations",
		`{"model":"painter"}`)
	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}

func TestImageGenerations_NoImageReturned(t *testing.T) {
	gw := &stubGateway{invoke: func(context.Context, domain.InvokeRequest) (domain.InvokeResponse, error) {
		return domain.InvokeResponse{FinishReason: domain.FinishReasonComplete}, nil
	}}
	rec := do(t, newTestServer(t, gw), http.MethodPost, "/v1/images/generations",
		`{"model":"painter","prompt":"cat"}`)
	if rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", rec.Code)
	}
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings
// ---------------------------------------------------------------------------

func TestEmbeddings_SingleAndBatch(t *testing.T) {
	t.Run("single string", func(t *testing.T) {
		rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/embeddings",
			`{"model":"embedder","input":"hello"}`)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d (%s)", rec.Code, rec.Body.String())
		}
		var out embeddingResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if len(out.Data) != 1 || out.Data[0].Object != "embedding" {
			t.Fatalf("data = %+v", out.Data)
		}
		if len(out.Data[0].Embedding) != 2 {
			t.Errorf("embedding = %v", out.Data[0].Embedding)
		}
	})

	t.Run("batch preserves order", func(t *testing.T) {
		rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/embeddings",
			`{"model":"embedder","input":["a","b","c"]}`)
		if rec.Code != http.StatusOK {
			t.Fatalf("status = %d", rec.Code)
		}
		var out embeddingResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if len(out.Data) != 3 {
			t.Fatalf("got %d rows, want 3", len(out.Data))
		}
		for i, row := range out.Data {
			if row.Index != i {
				t.Errorf("row %d carries index %d", i, row.Index)
			}
		}
	})
}

func TestEmbeddings_Base64IsRefused(t *testing.T) {
	rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/embeddings",
		`{"model":"embedder","input":"hi","encoding_format":"base64"}`)
	if rec.Code != http.StatusNotImplemented {
		t.Errorf("status = %d, want 501", rec.Code)
	}
}

func TestEmbeddings_BadInput(t *testing.T) {
	for _, body := range []string{
		`{"model":"embedder"}`,
		`{"model":"embedder","input":""}`,
		`{"model":"embedder","input":[]}`,
		`{"model":"embedder","input":[1,2,3]}`, // a token array is not supported
		`{"input":"hi"}`,
	} {
		rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/embeddings", body)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %s: status = %d, want 400", body, rec.Code)
		}
	}
}

// ---------------------------------------------------------------------------
// Probes and error envelope
// ---------------------------------------------------------------------------

func TestProbes(t *testing.T) {
	s := newTestServer(t, &stubGateway{})
	for _, path := range []string{"/healthz", "/readyz"} {
		if rec := do(t, s, http.MethodGet, path, ""); rec.Code != http.StatusOK {
			t.Errorf("%s = %d, want 200", path, rec.Code)
		}
	}
}

func TestErrorEnvelopeShape(t *testing.T) {
	rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/chat/completions",
		`{"model":"ghost","messages":[{"role":"user","content":"hi"}]}`)
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("content-type = %q", ct)
	}
	e := decodeError(t, rec)
	if e.Error.Message == "" || e.Error.Type == "" || e.Error.Code == "" {
		t.Errorf("envelope has an empty field: %+v", e.Error)
	}
}

func TestLegacyCompletionsPath(t *testing.T) {
	// Older clients post to /v1/completions; serving it is a one-line alias.
	rec := do(t, newTestServer(t, &stubGateway{}), http.MethodPost, "/v1/completions",
		`{"model":"chatty","messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Code)
	}
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

func TestUpstreamErrorTypeMapping(t *testing.T) {
	cases := map[int]string{
		http.StatusUnauthorized:        "authentication_error",
		http.StatusForbidden:           "authentication_error",
		http.StatusTooManyRequests:     "rate_limit_error",
		http.StatusNotFound:            "invalid_request_error",
		http.StatusInternalServerError: "upstream_error",
	}
	for status, want := range cases {
		if got := upstreamErrorType(status); got != want {
			t.Errorf("upstreamErrorType(%d) = %q, want %q", status, got, want)
		}
	}
}

func TestParseEmbeddingInput(t *testing.T) {
	got, err := parseEmbeddingInput(json.RawMessage(`"one"`))
	if err != nil || len(got) != 1 || got[0] != "one" {
		t.Errorf("single string = %v, %v", got, err)
	}
	got, err = parseEmbeddingInput(json.RawMessage(`["a","b"]`))
	if err != nil || len(got) != 2 {
		t.Errorf("array = %v, %v", got, err)
	}
	if _, err := parseEmbeddingInput(nil); err == nil {
		t.Error("nil input should error")
	}
	if _, err := parseEmbeddingInput(json.RawMessage(`[1,2]`)); err == nil {
		t.Error("a numeric array is a token array and must be refused")
	}
}

// writeTempRegistry materialises a registry file for a test.
func writeTempRegistry(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "models.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write registry: %v", err)
	}
	return path
}
