package httpapi

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/config"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// mustRegistry parses a registry for a test.
func mustRegistry(t *testing.T, body string) *config.Registry {
	t.Helper()
	reg, err := config.LoadRegistry(writeTempRegistry(t, body))
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	return reg
}

// groundedRegistry carries one entry per grounding shape plus one that cannot
// ground at all, so the gate has something to refuse.
func groundedRegistry(t *testing.T) string {
	t.Helper()
	return `
models:
  - id: searcher
    provider: openai
    base_url: https://api.example.test/v1
    capabilities: [chat, web_search]
    grounding:
      surface: responses

  - id: chatsearcher
    provider: openai
    base_url: https://api.example.test/v1
    capabilities: [chat, web_search]
    grounding:
      surface: chat_completions

  - id: claude-searcher
    provider: anthropic
    base_url: https://api.example.test/v1
    capabilities: [chat, web_search]
    grounding:
      surface: messages

  - id: plain
    provider: openai
    base_url: https://api.example.test/v1
    capabilities: [chat]
`
}

func newGroundedServer(t *testing.T, gw Gateway) *Server {
	t.Helper()
	s, err := NewServer(Config{
		Gateway:         gw,
		Models:          mustRegistry(t, groundedRegistry(t)),
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

func groundedResponse() domain.InvokeResponse {
	return domain.InvokeResponse{
		InvocationID:     "inv-grounded",
		Completion:       "Max Verstappen won the most recent race.",
		ModelVersion:     "gpt-5",
		Vendor:           "openai",
		Usage:            domain.TokenUsage{InputTokens: 25, OutputTokens: 12},
		FinishReason:     domain.FinishReasonComplete,
		CompletedAt:      time.Unix(1_700_000_000, 0),
		FallbackChain:    []string{"openai:gpt-5"},
		GroundingSurface: "responses",
		Citations: []domain.GroundingCitation{
			{URL: "https://f1.example/race-report", Title: "Race report", StartIndex: 0, EndIndex: 3},
			{URL: "https://f1.example/standings", Title: "Standings"},
		},
		SearchQueries: []string{"most recent Formula 1 race winner"},
	}
}

func doResponses(t *testing.T, s *Server, body string) *httptest.ResponseRecorder {
	t.Helper()
	return do(t, s, http.MethodPost, "/v1/responses", body)
}

func doRaw(t *testing.T, s *Server, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	return do(t, s, method, path, body)
}

// ---------------------------------------------------------------------------
// POST /v1/responses
// ---------------------------------------------------------------------------

// TestResponses_GroundedRequest mirrors the shape the caller in the original
// question sent, and is the load-bearing test for the whole feature.
func TestResponses_GroundedRequest(t *testing.T) {
	gw := &stubGateway{invoke: func(context.Context, domain.InvokeRequest) (domain.InvokeResponse, error) {
		return groundedResponse(), nil
	}}
	s := newGroundedServer(t, gw)

	rec := doResponses(t, s, `{
		"model": "searcher",
		"input": "Who won the most recent Formula 1 race?",
		"tools": [{"type": "web_search"}]
	}`)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200 (%s)", rec.Code, rec.Body.String())
	}

	// The grounding request must have reached the domain as the GROUNDED
	// modality, not as a plain text call.
	if gw.lastInvoke.ResponseModality != domain.ModalityGrounded {
		t.Errorf("modality = %q, want %q", gw.lastInvoke.ResponseModality, domain.ModalityGrounded)
	}
	if gw.lastInvoke.Prompt != "Who won the most recent Formula 1 race?" {
		t.Errorf("prompt = %q", gw.lastInvoke.Prompt)
	}

	var out responsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Object != "response" {
		t.Errorf("object = %q", out.Object)
	}
	if out.Model != "gpt-5" {
		t.Errorf("model = %q", out.Model)
	}
	if len(out.Output) != 2 {
		t.Fatalf("output items = %d, want a web_search_call plus a message", len(out.Output))
	}
	// Output is parsed BY TYPE by the client, so the types must be right.
	if out.Output[0]["type"] != "web_search_call" {
		t.Errorf("output[0].type = %v", out.Output[0]["type"])
	}
	if out.Output[1]["type"] != "message" {
		t.Errorf("output[1].type = %v", out.Output[1]["type"])
	}
	if out.Usage == nil || out.Usage.TotalTokens != 37 {
		t.Errorf("usage = %+v, want 25+12", out.Usage)
	}
}

func TestResponses_CitationsAppearBothNativelyAndNormalised(t *testing.T) {
	s := newGroundedServer(t, &stubGateway{invoke: func(context.Context, domain.InvokeRequest) (domain.InvokeResponse, error) {
		return groundedResponse(), nil
	}})
	rec := doResponses(t, s, `{"model":"searcher","input":"who won","tools":[{"type":"web_search"}]}`)

	var out responsesResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}

	// (a) Natively, as url_citation annotations on the output text — so an
	// unmodified OpenAI Responses client sees them where it expects.
	content, _ := out.Output[1]["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("content parts = %d", len(content))
	}
	part, _ := content[0].(map[string]any)
	annotations, _ := part["annotations"].([]any)
	if len(annotations) != 2 {
		t.Fatalf("annotations = %d, want 2", len(annotations))
	}
	first, _ := annotations[0].(map[string]any)
	if first["type"] != "url_citation" || first["url"] != "https://f1.example/race-report" {
		t.Errorf("annotation[0] = %v", first)
	}

	// (b) Normalised under the gateway's key, so a client does not need a
	// branch per provider.
	if out.ChoraGateway == nil {
		t.Fatal("chora_gateway metadata is missing")
	}
	if !out.ChoraGateway.Grounded {
		t.Error("chora_gateway.grounded = false")
	}
	if len(out.ChoraGateway.Citations) != 2 {
		t.Errorf("normalised citations = %+v, want 2", out.ChoraGateway.Citations)
	}
	if out.ChoraGateway.GroundingSurface != "responses" {
		t.Errorf("grounding surface = %q", out.ChoraGateway.GroundingSurface)
	}
	if len(out.ChoraGateway.SearchQueries) != 1 {
		t.Errorf("search queries = %v", out.ChoraGateway.SearchQueries)
	}
}

func TestResponses_NoToolsIsAPlainCall(t *testing.T) {
	gw := &stubGateway{}
	rec := doResponses(t, newGroundedServer(t, gw), `{"model":"searcher","input":"just answer"}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d (%s)", rec.Code, rec.Body.String())
	}
	// A caller who does not ask for search must not get it: grounding costs
	// money and changes the answer.
	if gw.lastInvoke.ResponseModality == domain.ModalityGrounded {
		t.Error("a plain call was silently grounded")
	}
}

func TestResponses_NonSearchToolsDoNotGround(t *testing.T) {
	gw := &stubGateway{}
	rec := doResponses(t, newGroundedServer(t, gw),
		`{"model":"searcher","input":"hi","tools":[{"type":"function","function":{"name":"lookup"}}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d (%s)", rec.Code, rec.Body.String())
	}
	if gw.lastInvoke.ResponseModality == domain.ModalityGrounded {
		t.Error("a function tool was mistaken for web search")
	}
}

func TestResponses_GroundingRefusedOnANonGroundedModel(t *testing.T) {
	// The refusal that matters: an ungrounded answer presented as grounded is
	// worse than an error.
	//
	// Asserted at the facade because the gate is enforced HERE, before any
	// dispatch. The domain enforces it too (see the domain tests), but a
	// caller reaching the domain at all means a wasted round trip and a 500
	// instead of an actionable 400.
	gw := &stubGateway{}
	rec := doResponses(t, newGroundedServer(t, gw),
		`{"model":"plain","input":"who won","tools":[{"type":"web_search"}]}`)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (%s)", rec.Code, rec.Body.String())
	}
	e := decodeError(t, rec)
	if e.Error.Code != "grounding_unavailable" {
		t.Errorf("code = %q, want grounding_unavailable", e.Error.Code)
	}
	if !strings.Contains(e.Error.Message, "web_search") {
		t.Errorf("message should name the missing capability: %q", e.Error.Message)
	}
	if gw.invokeN != 0 {
		t.Error("the refused request still reached the provider")
	}
}

func TestResponses_WebFetchAlsoCountsAsGrounding(t *testing.T) {
	gw := &stubGateway{}
	doResponses(t, newGroundedServer(t, gw),
		`{"model":"searcher","input":"what is this page","tools":[{"type":"web_fetch"}]}`)
	if gw.lastInvoke.ResponseModality != domain.ModalityGrounded {
		t.Error("web_fetch should ground too; it is the same egress")
	}
}

func TestResponses_StreamIsRefused(t *testing.T) {
	gw := &stubGateway{}
	rec := doResponses(t, newGroundedServer(t, gw),
		`{"model":"searcher","input":"hi","stream":true}`)
	if rec.Code != http.StatusNotImplemented {
		t.Errorf("status = %d, want 501", rec.Code)
	}
	if gw.invokeN != 0 {
		t.Error("a refused streaming request still spent a search")
	}
}

func TestResponses_WrongMethod(t *testing.T) {
	rec := doRaw(t, newGroundedServer(t, &stubGateway{}), http.MethodGet, "/v1/responses", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestResponses_MissingFields(t *testing.T) {
	for _, body := range []string{
		`{"input":"hi","tools":[{"type":"web_search"}]}`,
		`{"model":"searcher","tools":[{"type":"web_search"}]}`,
		`{"model":"searcher","input":"","tools":[{"type":"web_search"}]}`,
		`{"model":"searcher"}`,
	} {
		rec := doResponses(t, newGroundedServer(t, &stubGateway{}), body)
		if rec.Code != http.StatusBadRequest {
			t.Errorf("body %s: status = %d, want 400", body, rec.Code)
		}
	}
}

func TestResponses_UnknownModelIs404(t *testing.T) {
	rec := doResponses(t, newGroundedServer(t, &stubGateway{}),
		`{"model":"ghost","input":"hi","tools":[{"type":"web_search"}]}`)
	if rec.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", rec.Code)
	}
}

func TestResponses_BudgetBlockIs402(t *testing.T) {
	gw := &stubGateway{invoke: func(context.Context, domain.InvokeRequest) (domain.InvokeResponse, error) {
		return domain.InvokeResponse{
			InvocationID: "inv-1",
			FinishReason: domain.FinishReasonBudgetBlock,
			FinishDetail: "tenant LLM budget exhausted; policy=block",
		}, nil
	}}
	rec := doResponses(t, newGroundedServer(t, gw),
		`{"model":"searcher","input":"hi","tools":[{"type":"web_search"}]}`)
	// A search that cannot be afforded must be refused BEFORE it is spent.
	if rec.Code != http.StatusPaymentRequired {
		t.Fatalf("status = %d, want 402", rec.Code)
	}
}

func TestResponses_BudgetBlockHappensBeforeAnySearch(t *testing.T) {
	// The gateway's budget gate runs before dispatch, so a 402 here means no
	// search request was ever made. Asserted explicitly because web search is
	// priced per request: a refused-but-searched call is real money.
	gw := &stubGateway{invoke: func(context.Context, domain.InvokeRequest) (domain.InvokeResponse, error) {
		return domain.InvokeResponse{FinishReason: domain.FinishReasonBudgetBlock, FinishDetail: "exhausted"}, nil
	}}
	rec := doResponses(t, newGroundedServer(t, gw),
		`{"model":"searcher","input":"hi","tools":[{"type":"web_search"}]}`)
	if rec.Code != http.StatusPaymentRequired {
		t.Fatalf("status = %d", rec.Code)
	}
	if gw.lastInvoke.ResponseModality != domain.ModalityGrounded {
		t.Error("the request was not even marked as grounded")
	}
}

// ---------------------------------------------------------------------------
// Input normalisation
// ---------------------------------------------------------------------------

func TestFlattenResponsesInput(t *testing.T) {
	cases := []struct {
		name        string
		in          string
		wantPrompt  string
		wantContent bool
		wantErr     bool
	}{
		{name: "bare string", in: `"hello"`, wantPrompt: "hello"},
		{name: "single item collapses to a prompt", in: `[{"role":"user","content":"hello"}]`, wantPrompt: "hello"},
		{name: "item with a bare text field", in: `[{"role":"user","text":"hello"}]`, wantPrompt: "hello"},
		{name: "item with content parts", in: `[{"role":"user","content":[{"type":"input_text","text":"a"},{"type":"input_text","text":"b"}]}]`, wantPrompt: "ab"},
		{name: "two items become a conversation", in: `[{"role":"user","content":"one"},{"role":"assistant","content":"two"}]`, wantContent: true},
		{name: "empty string", in: `""`, wantErr: true},
		{name: "empty array", in: `[]`, wantErr: true},
		{name: "null", in: `null`, wantErr: true},
		{name: "non-text part", in: `[{"role":"user","content":[{"type":"input_image","image_url":"x"}]}]`, wantErr: true},
		{name: "wrong type", in: `42`, wantErr: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var input any
			if err := json.Unmarshal([]byte(tc.in), &input); err != nil {
				t.Fatalf("fixture is not valid json: %v", err)
			}
			prompt, contents, err := flattenResponsesInput(input)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected an error, got prompt=%q contents=%q", prompt, contents)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.wantContent {
				if contents == "" {
					t.Errorf("want a conversation, got prompt=%q", prompt)
				}
				return
			}
			if prompt != tc.wantPrompt {
				t.Errorf("prompt = %q, want %q", prompt, tc.wantPrompt)
			}
		})
	}
}

func TestResponses_ConversationReachesTheDomain(t *testing.T) {
	gw := &stubGateway{}
	doResponses(t, newGroundedServer(t, gw), `{
		"model":"searcher",
		"input":[{"role":"user","content":"one"},{"role":"assistant","content":"two"},{"role":"user","content":"three"}],
		"tools":[{"type":"web_search"}]
	}`)
	if !strings.Contains(gw.lastInvoke.ContentsJSON, `"three"`) {
		t.Errorf("conversation lost: %q", gw.lastInvoke.ContentsJSON)
	}
}

func TestResponses_InstructionsBecomeTheSystemPrompt(t *testing.T) {
	gw := &stubGateway{}
	doResponses(t, newGroundedServer(t, gw),
		`{"model":"searcher","input":"hi","instructions":"be terse","tools":[{"type":"web_search"}]}`)
	if gw.lastInvoke.SystemPrompt != "be terse" {
		t.Errorf("system prompt = %q", gw.lastInvoke.SystemPrompt)
	}
}

func TestResponses_GenerationConfigForwarded(t *testing.T) {
	gw := &stubGateway{}
	doResponses(t, newGroundedServer(t, gw),
		`{"model":"searcher","input":"hi","max_output_tokens":1500,"reasoning_effort":"high","tools":[{"type":"web_search"}]}`)
	cfg := gw.lastInvoke.GenerationConfig
	if cfg["max_output_tokens"] != float64(1500) {
		t.Errorf("max_output_tokens = %v", cfg["max_output_tokens"])
	}
	if cfg["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v", cfg["reasoning_effort"])
	}
}

func TestResponses_Attribution(t *testing.T) {
	gw := &stubGateway{}
	req := httptest.NewRequest(http.MethodPost, "/v1/responses",
		strings.NewReader(`{"model":"searcher","input":"hi","user":"alice","tools":[{"type":"web_search"}]}`))
	req.Header.Set("X-Chora-Tenant-Id", "tenant-7")
	rec := httptest.NewRecorder()
	newGroundedServer(t, gw).ServeHTTP(rec, req)
	if gw.lastInvoke.TenantID != "tenant-7" {
		t.Errorf("tenant = %q", gw.lastInvoke.TenantID)
	}
	if gw.lastInvoke.ActionCode != "alice" {
		t.Errorf("action code = %q", gw.lastInvoke.ActionCode)
	}
}

// ---------------------------------------------------------------------------
// Chat-completions grounding
// ---------------------------------------------------------------------------

func TestChatCompletions_GroundingHonouredOnTheConfiguredSurface(t *testing.T) {
	gw := &stubGateway{}
	rec := doRaw(t, newGroundedServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"chatsearcher","messages":[{"role":"user","content":"who won"}],"tools":[{"type":"web_search"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d (%s)", rec.Code, rec.Body.String())
	}
	if gw.lastInvoke.ResponseModality != domain.ModalityGrounded {
		t.Errorf("modality = %q, want GROUNDED", gw.lastInvoke.ResponseModality)
	}
}

func TestChatCompletions_GroundingRefusedOnTheWrongSurface(t *testing.T) {
	// "searcher" grounds on /v1/responses. OpenAI does NOT offer hosted search
	// on chat completions, so a grounding request there must be redirected, not
	// dispatched with a tool the provider ignores.
	gw := &stubGateway{}
	rec := doRaw(t, newGroundedServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"searcher","messages":[{"role":"user","content":"who won"}],"tools":[{"type":"web_search"}]}`)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (%s)", rec.Code, rec.Body.String())
	}
	e := decodeError(t, rec)
	if e.Error.Code != "wrong_grounding_surface" {
		t.Errorf("code = %q", e.Error.Code)
	}
	if !strings.Contains(e.Error.Message, "/v1/responses") {
		t.Errorf("message should point at the right surface: %q", e.Error.Message)
	}
	if gw.invokeN != 0 {
		t.Error("the misdirected request still reached the provider")
	}
}

func TestChatCompletions_GroundingRefusedOnANonGroundedModel(t *testing.T) {
	gw := &stubGateway{}
	rec := doRaw(t, newGroundedServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"plain","messages":[{"role":"user","content":"who won"}],"tools":[{"type":"web_search"}]}`)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	if e := decodeError(t, rec); e.Error.Code != "grounding_unavailable" {
		t.Errorf("code = %q", e.Error.Code)
	}
}

func TestChatCompletions_PlainCallUnaffectedByGroundingCapability(t *testing.T) {
	// A model that CAN search must still serve ordinary calls without any
	// ceremony.
	gw := &stubGateway{}
	rec := doRaw(t, newGroundedServer(t, gw), http.MethodPost, "/v1/chat/completions",
		`{"model":"searcher","messages":[{"role":"user","content":"hi"}]}`)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d (%s)", rec.Code, rec.Body.String())
	}
	if gw.lastInvoke.ResponseModality == domain.ModalityGrounded {
		t.Error("a plain call was silently grounded")
	}
}

func TestChatCompletions_CitationsSurfaced(t *testing.T) {
	s := newGroundedServer(t, &stubGateway{invoke: func(context.Context, domain.InvokeRequest) (domain.InvokeResponse, error) {
		return groundedResponse(), nil
	}})
	rec := doRaw(t, s, http.MethodPost, "/v1/chat/completions",
		`{"model":"chatsearcher","messages":[{"role":"user","content":"who won"}],"tools":[{"type":"web_search"}]}`)

	var out chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Gateway == nil || !out.Gateway.Grounded {
		t.Fatalf("gateway meta = %+v", out.Gateway)
	}
	if len(out.Gateway.Citations) != 2 {
		t.Errorf("citations = %+v, want 2", out.Gateway.Citations)
	}
}

func TestToolNamesWebSearch(t *testing.T) {
	cases := map[string]bool{
		`[{"type":"web_search"}]`:            true,
		`[{"type":"web_search_preview"}]`:    true,
		`[{"type":"web_search_20250305"}]`:   true,
		`[{"type":"web_search_2025_08_26"}]`: true,
		`[{"type":"web_fetch"}]`:             true,
		`[{"type":"web_fetch_20260318"}]`:    true,
		`[{"type":"WEB_SEARCH"}]`:            true,
		`[{"name":"web_search"}]`:            true,
		`[{"type":"function"}]`:              false,
		`[{"type":"code_interpreter"}]`:      false,
		`[]`:                                 false,
		`[{"type":"file_search"}]`:           false,
	}
	for body, want := range cases {
		var tools []json.RawMessage
		if err := json.Unmarshal([]byte(body), &tools); err != nil {
			t.Fatalf("fixture %s: %v", body, err)
		}
		if got := toolNamesWebSearch(tools); got != want {
			t.Errorf("toolNamesWebSearch(%s) = %v, want %v", body, got, want)
		}
	}
}

func TestGroundingSurfaceFor(t *testing.T) {
	s := newGroundedServer(t, &stubGateway{})

	cases := []struct {
		model  string
		want   domain.GroundingSurface
		errHas bool
	}{
		{model: "searcher", want: domain.SurfaceResponses},
		{model: "chatsearcher", want: domain.SurfaceChatCompletions},
		{model: "claude-searcher", want: domain.SurfaceMessages},
		{model: "plain", errHas: true},
		{model: "ghost", errHas: true},
	}
	for _, tc := range cases {
		got, err := s.groundingSurfaceFor(tc.model)
		if tc.errHas {
			if err == nil {
				t.Errorf("groundingSurfaceFor(%q) = %q, want an error", tc.model, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("groundingSurfaceFor(%q): %v", tc.model, err)
			continue
		}
		if got != tc.want {
			t.Errorf("groundingSurfaceFor(%q) = %q, want %q", tc.model, got, tc.want)
		}
	}
}

func TestBuildResponsesOutput_NoGroundingMeansNoSearchCallItem(t *testing.T) {
	items := buildResponsesOutput(domain.InvokeResponse{Completion: "plain"}, domain.ModalityText)
	if len(items) != 1 {
		t.Fatalf("output items = %d, want just the message", len(items))
	}
	if items[0]["type"] != "message" {
		t.Errorf("items[0].type = %v", items[0]["type"])
	}
}

func TestBuildResponsesOutput_EmptyAnnotationsOnAnUncitedAnswer(t *testing.T) {
	// A grounded call that found nothing citable must still emit an empty
	// annotations array rather than omitting the key, so a client can tell it
	// apart from a non-grounded call.
	items := buildResponsesOutput(domain.InvokeResponse{
		InvocationID: "inv-1",
		Completion:   "I could not find anything citable.",
	}, domain.ModalityGrounded)

	content, _ := items[1]["content"].([]map[string]any)
	if len(content) == 0 {
		t.Fatalf("no content parts: %v", items[1])
	}
	part := content[0]
	ann, present := part["annotations"]
	if !present {
		t.Fatal("annotations key is absent; a client cannot distinguish this from a non-grounded call")
	}
	if l, _ := ann.([]any); len(l) != 0 {
		t.Errorf("annotations = %v, want empty", ann)
	}
}

// TestBuildResponsesOutput_EmitsOneMessageItemPerMessage is the facade-side
// guard: a reasoning model's narration and answer must stay separate items, so
// a Responses client can tell them apart and citations attach to the right one.
func TestBuildResponsesOutput_EmitsOneMessageItemPerMessage(t *testing.T) {
	items := buildResponsesOutput(domain.InvokeResponse{
		InvocationID: "inv-1",
		Completion:   "6.21 million.",
		Messages: []domain.GroundingMessage{
			{Text: "I'll look that up."},
			{Text: "The results conflict."},
			{Text: "6.21 million.", Citations: []domain.GroundingCitation{
				{URL: "https://population.gov.sg/pib", Title: "Population in Brief", StartIndex: 0, EndIndex: 4},
			}},
		},
	}, domain.ModalityGrounded)

	// One search item plus three message items.
	if len(items) != 4 {
		t.Fatalf("output items = %d, want 4", len(items))
	}
	for i := 1; i < 4; i++ {
		if items[i]["type"] != "message" {
			t.Errorf("items[%d].type = %v, want message", i, items[i]["type"])
		}
	}
	// Only the answer carries annotations.
	last := items[3]["content"].([]map[string]any)[0]
	anns := last["annotations"].([]map[string]any)
	if len(anns) != 1 {
		t.Fatalf("the answer's annotations = %v, want 1", anns)
	}
	if anns[0]["url"] != "https://population.gov.sg/pib" {
		t.Errorf("annotation url = %v", anns[0]["url"])
	}
	narration := items[1]["content"].([]map[string]any)[0]
	if l, _ := narration["annotations"].([]map[string]any); len(l) != 0 {
		t.Errorf("narration was given citations: %v", l)
	}
}

func TestAnnotationsToJSON_OmitsAbsentOffsets(t *testing.T) {
	// Anthropic supplies no character offsets. Emitting 0..0 would read as
	// "cited at the very start of the answer", which is a false statement.
	got := annotationsToJSON([]domain.GroundingCitation{
		{URL: "https://a.example", Title: "A"},
		{URL: "https://b.example", StartIndex: 4, EndIndex: 9},
	})
	if _, present := got[0]["start_index"]; present {
		t.Errorf("absent offsets were emitted: %v", got[0])
	}
	if got[1]["start_index"] != 4 || got[1]["end_index"] != 9 {
		t.Errorf("real offsets lost: %v", got[1])
	}
	if got[0]["type"] != "url_citation" || got[0]["url"] != "https://a.example" {
		t.Errorf("annotation shape = %v", got[0])
	}
}

func TestAnnotationsToJSON_EmptyStaysAnArray(t *testing.T) {
	got := annotationsToJSON(nil)
	if got == nil {
		t.Fatal("annotations must serialise as [] not null, so a client can tell " +
			"'grounded, nothing citable' from 'not a grounded call'")
	}
	if len(got) != 0 {
		t.Errorf("annotations = %v, want empty", got)
	}
}
