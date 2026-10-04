package anthropic

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/config"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// ---------------------------------------------------------------------------
// Stub upstream
//
// The OpenAI adapter's suite already carries a `stubUpstream`, but it lives in
// package openai and unexported helpers cannot cross a package boundary. This is
// a local copy on purpose: sharing it would mean exporting test scaffolding into
// a non-test file, which would put it in the production build.
// ---------------------------------------------------------------------------

// recordedRequest is one request the fake provider received.
type recordedRequest struct {
	Method string
	Path   string
	Header http.Header
	Body   map[string]any
}

// stubUpstream is a fake provider. ONE handler serves every test: it records
// the request, then delegates to whatever response the test installed, so the
// recorder cannot be accidentally replaced along with the responder.
type stubUpstream struct {
	*httptest.Server

	mu       sync.Mutex
	requests []recordedRequest
	respond  func(w http.ResponseWriter, r *http.Request)
}

// respondJSON installs a canned JSON response.
func (u *stubUpstream) respondJSON(payload string) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.respond = func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(payload))
	}
}

// respondFunc installs an arbitrary handler.
func (u *stubUpstream) respondFunc(f func(w http.ResponseWriter, r *http.Request)) {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.respond = f
}

// last returns the most recent recorded request.
func (u *stubUpstream) last(t *testing.T) recordedRequest {
	t.Helper()
	u.mu.Lock()
	defer u.mu.Unlock()
	if len(u.requests) == 0 {
		t.Fatal("the stub upstream received no request")
	}
	return u.requests[len(u.requests)-1]
}

// count returns how many requests arrived.
func (u *stubUpstream) count() int {
	u.mu.Lock()
	defer u.mu.Unlock()
	return len(u.requests)
}

func newTestClient(t *testing.T, cfg Config) (*Client, *stubUpstream) {
	t.Helper()

	up := &stubUpstream{}
	up.respondJSON(`{}`)
	up.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		up.mu.Lock()
		up.requests = append(up.requests, recordedRequest{
			Method: r.Method,
			Path:   r.URL.Path,
			Header: r.Header.Clone(),
			Body:   body,
		})
		respond := up.respond
		up.mu.Unlock()
		respond(w, r)
	}))
	t.Cleanup(up.Close)

	if cfg.HTTPClient == nil {
		cfg.HTTPClient = up.Client()
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c, up
}

func writeRegistryFile(t *testing.T, body string) *config.Registry {
	t.Helper()
	path := t.TempDir() + "/models.yaml"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write registry: %v", err)
	}
	reg, err := config.LoadRegistry(path)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	return reg
}

// target is an Anthropic-shaped destination. MaxOutputTokens is set because a
// zero ceiling is exercised separately — it is the case that proves max_tokens
// is never omitted.
func target(base string) domain.TargetModel {
	return domain.TargetModel{
		Vendor:              domain.VendorFamilyAnthropic,
		LogicalModelID:      "m",
		UpstreamModel:       "claude-sonnet-4",
		BaseURL:             base,
		MessagesPath:        "/v1/messages",
		MaxOutputTokens:     512,
		Capabilities:        []string{domain.CapabilityChat},
		ExtraHeaders:        nil,
		ContextWindow:       200000,
		APIKeyRef:           "ANTHROPIC_API_KEY",
		ChatCompletionsPath: "", // Anthropic entries must not need this
	}
}

func TestFamily(t *testing.T) {
	c, _ := newTestClient(t, Config{})
	if c.Family() != domain.VendorFamilyAnthropic {
		t.Errorf("Family = %q", c.Family())
	}
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

func TestGenerate_ChatHappyPath(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{
		"model":"claude-sonnet-4-20250514",
		"stop_reason":"end_turn",
		"content":[{"type":"text","text":"the answer"}],
		"usage":{"input_tokens":12,"output_tokens":4}
	}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:     target(up.URL),
		Prompt:     "hi",
		Credential: "sk-ant-test",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Completion != "the answer" {
		t.Errorf("completion = %q", resp.Completion)
	}
	if resp.Usage.InputTokens != 12 || resp.Usage.OutputTokens != 4 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.FinishReason != domain.FinishReasonComplete {
		t.Errorf("finish = %v", resp.FinishReason)
	}
	// The upstream echoes the pinned model version, which is the useful answer
	// to "which model actually ran" when the registry aliased a name onto it.
	if resp.ModelVersion != "claude-sonnet-4-20250514" {
		t.Errorf("model version = %q", resp.ModelVersion)
	}

	got := up.last(t)
	if got.Method != http.MethodPost {
		t.Errorf("method = %q", got.Method)
	}
	if got.Path != "/v1/messages" {
		t.Errorf("path = %q, want /v1/messages", got.Path)
	}
	if got.Body["model"] != "claude-sonnet-4" {
		t.Errorf("upstream model sent = %v", got.Body["model"])
	}
}

func TestGenerate_UsageReportsCacheReadTokens(t *testing.T) {
	// Anthropic reports cache HITS separately from input_tokens. Folding them
	// together would overstate billable input on every cached turn.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{
		"content":[{"type":"text","text":"ok"}],
		"usage":{"input_tokens":1200,"output_tokens":80,"cache_read_input_tokens":900}
	}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Usage.CachedTokens != 900 {
		t.Errorf("cached tokens = %d, want 900", resp.Usage.CachedTokens)
	}
	if resp.Usage.InputTokens != 1200 {
		t.Errorf("input tokens = %d, want 1200", resp.Usage.InputTokens)
	}
}

func TestGenerate_AuthenticatesWithXAPIKeyNotBearer(t *testing.T) {
	// Anthropic rejects a bearer Authorization header outright, so sending one
	// (or sending the key in the wrong header) is a 401 on every call.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi", Credential: "sk-ant-secret",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	h := up.last(t).Header
	if got := h.Get("x-api-key"); got != "sk-ant-secret" {
		t.Errorf("x-api-key = %q, want the credential verbatim", got)
	}
	if _, present := h["Authorization"]; present {
		t.Error("an Authorization header was sent; Anthropic does not authenticate with a bearer token")
	}
}

func TestGenerate_SendsPinnedAPIVersionHeader(t *testing.T) {
	// Without anthropic-version the API rejects the request, and the version
	// is a protocol constant rather than a deployment setting — so it is
	// pinned in the adapter, not read from the environment.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := up.last(t).Header.Get("anthropic-version"); got != "2023-06-01" {
		t.Errorf("anthropic-version = %q, want the pinned 2023-06-01", got)
	}
}

func TestGenerate_APIVersionIsOverridable(t *testing.T) {
	// A gateway fronting an Anthropic-shaped server may need a different
	// version pinned, so the constant is a default rather than a hardwire.
	c, up := newTestClient(t, Config{APIVersion: "2025-03-19"})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := up.last(t).Header.Get("anthropic-version"); got != "2025-03-19" {
		t.Errorf("anthropic-version = %q, want the override", got)
	}
}

func TestGenerate_MaxTokensIsAlwaysPresent(t *testing.T) {
	// max_tokens is REQUIRED by the Anthropic API (unlike OpenAI's, where it
	// is optional) — omitting it is a 400 on every single call. The adapter
	// therefore always sends it, defaulting to a conservative ceiling when the
	// registry leaves max_output_tokens unset, so an incomplete registry entry
	// degrades rather than breaking.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	tgt := target(up.URL)
	tgt.MaxOutputTokens = 0 // a registry entry that never set the ceiling
	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: tgt, Prompt: "hi",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	body := up.last(t).Body
	v, present := body["max_tokens"]
	if !present {
		t.Fatalf("max_tokens omitted: %s", mustMarshal(body))
	}
	if v != float64(defaultMaxTokens) {
		t.Errorf("max_tokens = %v, want the %d default", v, defaultMaxTokens)
	}
}

func TestGenerate_MaxTokensUsesTheRegistryCeiling(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := up.last(t).Body["max_tokens"]; got != float64(512) {
		t.Errorf("max_tokens = %v, want the registry ceiling 512", got)
	}
}

func TestGenerate_SystemPromptGoesInTheSystemField(t *testing.T) {
	// The messages API has a top-level `system` field. A system ROLE inside
	// `messages` is accepted by the endpoint but is not the system prompt and
	// is silently weaker, so the separation must be preserved on the wire.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi", SystemPrompt: "be terse",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	body := up.last(t).Body
	if body["system"] != "be terse" {
		t.Errorf("system field = %v, want the system prompt", body["system"])
	}
	msgs, _ := body["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("messages = %v, want only the user turn", body["messages"])
	}
	for _, m := range msgs {
		role, _ := m.(map[string]any)["role"].(string)
		if role == "system" {
			t.Error("the system prompt was sent as a message; it belongs in the top-level system field")
		}
	}
}

func TestGenerate_NoSystemPromptOmitsTheField(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if _, present := up.last(t).Body["system"]; present {
		t.Error("an empty system field was sent")
	}
}

func TestGenerate_ContentsJSONBecomesTheConversation(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:       target(up.URL),
		Prompt:       "ignored",
		SystemPrompt: "be terse",
		ContentsJSON: `[{"role":"user","content":"first"},{"role":"assistant","content":"second"},` +
			`{"role":"user","content":"third"}]`,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	msgs, _ := up.last(t).Body["messages"].([]any)
	if len(msgs) != 3 {
		t.Fatalf("messages = %v, want the 3-turn conversation", up.last(t).Body["messages"])
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "user" || first["content"] != "first" {
		t.Errorf("first message = %v", first)
	}
}

func TestGenerate_MalformedContentsFallsBackToPrompt(t *testing.T) {
	// A malformed conversation is a caller bug, but dropping the whole call
	// would be worse than sending the bare prompt — the caller still gets an
	// answer and a visible (unparseable) record of what was sent.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "the real prompt", ContentsJSON: `{not json`,
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(mustMarshal(up.last(t).Body), "the real prompt") {
		t.Errorf("fallback did not carry the prompt: %s", mustMarshal(up.last(t).Body))
	}
}

func TestGenerate_EmptyContentsArrayFallsBackToPrompt(t *testing.T) {
	// An empty array is valid JSON but carries no turns; sending messages:[]
	// is a 400 on the messages API, so the prompt must win.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "the real prompt", ContentsJSON: `[]`,
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(mustMarshal(up.last(t).Body), "the real prompt") {
		t.Errorf("fallback did not carry the prompt: %s", mustMarshal(up.last(t).Body))
	}
}

func TestGenerate_StopSequencesFromEveryShape(t *testing.T) {
	// The generation config arrives either from JSON decoding (so stop is a
	// []any) or hand-built by the HTTP facade (a string, or a []string).
	// Dropping the sequences because of a Go type would silently change the
	// generation the caller asked for.
	cases := []struct {
		name string
		in   any
		want []string
	}{
		{"single string", "END", []string{"END"}},
		{"string slice", []string{"END", "HALT"}, []string{"END", "HALT"}},
		{"decoded slice", []any{"END", "HALT"}, []string{"END", "HALT"}},
		{"decoded slice with non-strings", []any{"END", 42, nil}, []string{"END"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c, up := newTestClient(t, Config{})
			up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

			if _, err := c.Generate(context.Background(), domain.VendorRequest{
				Target:           target(up.URL),
				Prompt:           "hi",
				GenerationConfig: map[string]any{"stop": tc.in},
			}); err != nil {
				t.Fatalf("Generate: %v", err)
			}
			got, _ := up.last(t).Body["stop_sequences"].([]any)
			if len(got) != len(tc.want) {
				t.Fatalf("stop_sequences = %v, want %v", got, tc.want)
			}
			for i := range got {
				if got[i] != tc.want[i] {
					t.Errorf("stop_sequences[%d] = %v, want %q", i, got[i], tc.want[i])
				}
			}
		})
	}
}

func TestGenerate_StopSequencesAliasIsHonoured(t *testing.T) {
	// "stop_sequences" is the Anthropic-native spelling; a caller who used it
	// must not have it silently dropped.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "hi",
		GenerationConfig: map[string]any{"stop_sequences": []string{"END"}},
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	got, _ := up.last(t).Body["stop_sequences"].([]any)
	if len(got) != 1 || got[0] != "END" {
		t.Errorf("stop_sequences = %v, want [END]", got)
	}
}

func TestGenerate_ForwardsSamplingKnobs(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL),
		Prompt: "hi",
		GenerationConfig: map[string]any{
			"temperature": 0.3,
			"top_p":       0.8,
		},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	body := up.last(t).Body
	if body["temperature"] != 0.3 || body["top_p"] != 0.8 {
		t.Errorf("sampling knobs = %s", mustMarshal(body))
	}
}

func TestGenerate_CallerMaxTokensOverridesTheCeiling(t *testing.T) {
	// The service layer already clamps to the registry ceiling, so whatever
	// reaches the adapter is the authoritative number and must be forwarded.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "hi",
		GenerationConfig: map[string]any{"max_tokens": float64(64)},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := up.last(t).Body["max_tokens"]; got != float64(64) {
		t.Errorf("max_tokens = %v, want the caller's 64", got)
	}
}

func TestGenerate_ForwardsTools(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:    target(up.URL),
		Prompt:    "hi",
		ToolsJSON: `[{"name":"lookup","description":"look a thing up","input_schema":{"type":"object"}}]`,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(mustMarshal(up.last(t).Body), "lookup") {
		t.Errorf("tools dropped: %s", mustMarshal(up.last(t).Body))
	}
}

func TestGenerate_MalformedToolsIsDroppedNotFatal(t *testing.T) {
	// A tools blob the caller mangled should degrade to a plain generation
	// rather than fail the call; the completion is still the useful output.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi", ToolsJSON: `{broken`,
	}); err != nil {
		t.Fatalf("a malformed tools blob should not fail the call: %v", err)
	}
}

func TestGenerate_ForwardsTraceHeadersAndExtras(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:      target(up.URL),
		Prompt:      "hi",
		Traceparent: "00-abc-def-01",
		Tracestate:  "vendor=1",
		ExtraHeaders: map[string]string{
			"anthropic-beta": "prompt-caching-2024-07-31",
		},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	h := up.last(t).Header
	if h.Get("traceparent") != "00-abc-def-01" {
		t.Errorf("traceparent = %q", h.Get("traceparent"))
	}
	if h.Get("tracestate") != "vendor=1" {
		t.Errorf("tracestate = %q", h.Get("tracestate"))
	}
	// Registry-sourced beta flags are how a beta feature is opted into, and they
	// must reach the wire or the call silently runs the stable behaviour.
	if h.Get("anthropic-beta") != "prompt-caching-2024-07-31" {
		t.Errorf("anthropic-beta = %q, want the registry header", h.Get("anthropic-beta"))
	}
}

func TestGenerate_TextBlocksAreConcatenatedAndOthersSkipped(t *testing.T) {
	// A tool_use block has no Text field; blindly concatenating every block
	// would append empty strings and, worse, any future block type's payload.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[
		{"type":"text","text":"part one "},
		{"type":"tool_use","id":"call_1","name":"lookup","input":{"q":"x"}},
		{"type":"text","text":"part two"}
	]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Completion != "part one part two" {
		t.Errorf("completion = %q, want the text blocks joined", resp.Completion)
	}
}

func TestGenerate_StopReasonMapping(t *testing.T) {
	// max_tokens is a truncated answer, and the caller has to be able to tell
	// that apart from a finished turn. tool_use is a complete turn: the agent
	// loop continues from there, it is not a truncation.
	c, up := newTestClient(t, Config{})
	cases := []struct {
		stop string
		want domain.FinishReason
	}{
		{"end_turn", domain.FinishReasonComplete},
		{"stop_sequence", domain.FinishReasonComplete},
		{"tool_use", domain.FinishReasonComplete},
		{"max_tokens", domain.FinishReasonMaxTokens},
		{"", domain.FinishReasonComplete},
		{"something_new", domain.FinishReasonComplete},
	}
	for _, tc := range cases {
		t.Run("stop_reason="+tc.stop, func(t *testing.T) {
			up.respondJSON(`{"content":[{"type":"text","text":"ok"}],"stop_reason":"` + tc.stop + `"}`)
			resp, err := c.Generate(context.Background(), domain.VendorRequest{
				Target: target(up.URL), Prompt: "hi",
			})
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			if resp.FinishReason != tc.want {
				t.Errorf("finish = %v, want %v", resp.FinishReason, tc.want)
			}
		})
	}
}

func TestGenerate_MissingModelEchoFallsBackToTheTarget(t *testing.T) {
	// Some Anthropic-compatible servers omit `model`. The dispatched name is
	// the useful answer, so the ledger row is still attributable.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.ModelVersion != "claude-sonnet-4" {
		t.Errorf("model version = %q, want the dispatched upstream model", resp.ModelVersion)
	}
}

func TestGenerate_UpstreamErrorCarriesStatus(t *testing.T) {
	// The HTTP facade relays the provider's own status instead of flattening
	// every failure to 502: a 401 is the caller's credential problem and a 429
	// is a retryable rate limit, and neither is the gateway's fault.
	c, up := newTestClient(t, Config{})
	up.respondFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid x-api-key"}}`))
	})

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi", Credential: "sk-ant-bad",
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	upErr, ok := err.(*UpstreamError)
	if !ok {
		t.Fatalf("error = %T, want *UpstreamError", err)
	}
	if upErr.UpstreamStatusCode() != http.StatusUnauthorized {
		t.Errorf("status = %d, want 401", upErr.UpstreamStatusCode())
	}
	if !strings.Contains(upErr.Body, "invalid x-api-key") {
		t.Errorf("body = %q, want the provider's message", upErr.Body)
	}
	if !strings.Contains(upErr.Error(), "/v1/messages") {
		t.Errorf("error text = %q, want it to name the endpoint", upErr.Error())
	}
}

func TestGenerate_RateLimitIsAlsoAnUpstreamError(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error"}}`))
	})

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	})
	upErr, ok := err.(*UpstreamError)
	if !ok {
		t.Fatalf("error = %T (%v), want *UpstreamError", err, err)
	}
	if upErr.UpstreamStatusCode() != http.StatusTooManyRequests {
		t.Errorf("status = %d, want 429", upErr.UpstreamStatusCode())
	}
}

func TestGenerate_UpstreamErrorBodyIsBounded(t *testing.T) {
	// A provider error page can be megabytes; the whole body must not end up in
	// an error string that then goes into a log line and a gRPC message.
	c, up := newTestClient(t, Config{})
	up.respondFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(strings.Repeat("x", 10_000)))
	})

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	})
	upErr, ok := err.(*UpstreamError)
	if !ok {
		t.Fatalf("error = %T, want *UpstreamError", err)
	}
	if len(upErr.Body) > 2048+len("…") {
		t.Errorf("body length = %d, want it truncated to ~2048", len(upErr.Body))
	}
}

func TestGenerate_MalformedUpstreamJSON(t *testing.T) {
	// An HTML error page served with a 200 (a misconfigured proxy) must not be
	// reported as a successful call with an empty completion.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`<html>not json</html>`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	})
	if err == nil {
		t.Fatal("expected a decode error")
	}
	if _, isUpstream := err.(*UpstreamError); isUpstream {
		t.Error("a decode failure on a 200 was reported as an upstream status error")
	}
}

func TestGenerate_UnreachableUpstream(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.Close()

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	}); err == nil {
		t.Fatal("expected a transport error")
	}
}

func TestGenerate_TruncatedBodyIsNotASuccess(t *testing.T) {
	// A connection dropped mid-body must not look like a completed answer: the
	// caller would render "" as the model's reply, and the ledger row would
	// record a successful dispatch with no output.
	c, up := newTestClient(t, Config{})
	up.respondFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Length", "400")
		_, _ = w.Write([]byte(`{"content":[{"type":"text","text":"half an ans`))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		panic(http.ErrAbortHandler)
	})

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	}); err == nil {
		t.Fatal("a truncated body was reported as a successful call")
	}
}

func TestGenerate_AnUnparseableTargetURLIsRefusedBeforeDispatch(t *testing.T) {
	// A base_url carrying a control character (a stray byte from a templated
	// deployment file) is not a URL. Failing here means the operator sees a
	// gateway-side error naming the target, instead of a confusing provider 4xx
	// or a dispatch that goes nowhere.
	c, up := newTestClient(t, Config{})

	tgt := target("https://api.example.test/\x7f")
	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: tgt, Prompt: "hi",
	})
	if err == nil {
		t.Fatal("expected an error for an unparseable target URL")
	}
	if !strings.Contains(err.Error(), "anthropic:") {
		t.Errorf("error = %q, want it namespaced to the adapter rather than the provider", err)
	}
	if up.count() != 0 {
		t.Errorf("an unparseable URL still produced %d request(s)", up.count())
	}
}

func TestGenerate_ImageModalityIsRefusedBeforeAnySpend(t *testing.T) {
	// The messages API has no image-generation surface. Failing here — before
	// a request goes out — is the difference between a clear error and a
	// billable 404 from the provider.
	c, up := newTestClient(t, Config{})

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "a cat",
		ResponseModality: domain.ModalityImage,
	})
	if err == nil {
		t.Fatal("expected a refusal for response_modality=IMAGE")
	}
	if !strings.Contains(err.Error(), "image") {
		t.Errorf("error = %q, want it to name the image modality", err)
	}
	if !strings.Contains(err.Error(), string(target("").LogicalModelID)) {
		t.Errorf("error = %q, want it to name the model that cannot serve it", err)
	}
	if up.count() != 0 {
		t.Errorf("the refused request still reached the provider %d time(s)", up.count())
	}
}

func TestGenerate_ImageModalityIsRefusedCaseInsensitively(t *testing.T) {
	// The gRPC surface passes the string through verbatim, so a lowercase
	// "image" must not slip past the gate that "IMAGE" hits.
	c, up := newTestClient(t, Config{})

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "a cat",
		ResponseModality: "image",
	}); err == nil {
		t.Fatal("expected a refusal for a lowercase image modality")
	}
	if up.count() != 0 {
		t.Errorf("the refused request still reached the provider %d time(s)", up.count())
	}
}

func TestGenerate_TextModalityIsDispatched(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi", ResponseModality: domain.ModalityText,
	}); err != nil {
		t.Fatalf("an explicit TEXT modality must be dispatched, not refused: %v", err)
	}
}

func TestGenerate_UsesTheRegistryMessagesPath(t *testing.T) {
	// A gateway proxying Anthropic-shaped traffic at a non-standard path relies
	// on messages_path being honoured; the default /v1/messages is only a
	// fallback for an entry that leaves it blank.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	tgt := target(up.URL)
	tgt.MessagesPath = "/anthropic/v1/messages"
	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: tgt, Prompt: "hi",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := up.last(t).Path; got != "/anthropic/v1/messages" {
		t.Errorf("path = %q, want the registry-configured messages path", got)
	}
}

func TestGenerate_MessagesPathDefaultsToV1Messages(t *testing.T) {
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}]}`)

	tgt := target(up.URL)
	tgt.MessagesPath = ""
	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: tgt, Prompt: "hi",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := up.last(t).Path; got != "/v1/messages" {
		t.Errorf("path = %q, want the /v1/messages default", got)
	}
}

// ---------------------------------------------------------------------------
// Cost
// ---------------------------------------------------------------------------

func TestCostMicros(t *testing.T) {
	reg := writeRegistryFile(t, `
models:
  - id: priced
    provider: anthropic
    base_url: https://api.anthropic.com
    pricing:
      input_per_mtok_usd_micros: 1000000
      output_per_mtok_usd_micros: 2000000
      cached_per_mtok_usd_micros: 100000
  - id: free
    provider: anthropic
    base_url: https://api.anthropic.com
`)
	c, err := New(Config{Pricing: reg})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// 1000 in (500 of them cached) + 2000 out at $1 / $2 / $0.10 per Mtok.
	// The cached half is billed at the cache rate, NOT the input rate, so the
	// billable input is 500 not 1000.
	got := c.costMicros("priced", 1000, 2000, 500, 0)
	want := int64(500) + int64(4000) + int64(50)
	if got != want {
		t.Errorf("cost = %d, want %d", got, want)
	}

	// An unpriced model costs zero: the honest answer for a self-hosted or
	// operator-supplied model.
	if v := c.costMicros("free", 1000, 2000, 0, 0); v != 0 {
		t.Errorf("unpriced cost = %d, want 0", v)
	}
	if v := c.costMicros("ghost", 1000, 2000, 0, 0); v != 0 {
		t.Errorf("unknown model cost = %d, want 0", v)
	}

	// No registry at all is zero rather than a panic.
	bare, err := New(Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if v := bare.costMicros("priced", 1000, 2000, 0, 0); v != 0 {
		t.Errorf("nil-registry cost = %d, want 0", v)
	}
}

func TestCostMicros_CachedTokensAreNotDoubleCounted(t *testing.T) {
	// Cached input is a SUBSET of input; billing it at both rates would
	// overcharge every cached request.
	reg := writeRegistryFile(t, `
models:
  - id: m
    provider: anthropic
    base_url: https://api.anthropic.com
    pricing:
      input_per_mtok_usd_micros: 1000000
      output_per_mtok_usd_micros: 0
      cached_per_mtok_usd_micros: 0
`)
	c, _ := New(Config{Pricing: reg})
	if v := c.costMicros("m", 1000, 0, 1000, 0); v != 0 {
		t.Errorf("cost = %d, want 0 (the whole input was cached)", v)
	}
	// A nonsensical cached > input must not produce a negative debit.
	if v := c.costMicros("m", 100, 0, 500, 0); v != 0 {
		t.Errorf("cost = %d, want 0 rather than a negative bill", v)
	}
}

func TestGenerate_CostIsAttachedFromTheRegistry(t *testing.T) {
	// The cost flows into the budget debit and the ledger row, so a missing
	// attachment here means the tenant is never charged.
	reg := writeRegistryFile(t, `
models:
  - id: m
    provider: anthropic
    base_url: https://api.anthropic.com
    pricing:
      input_per_mtok_usd_micros: 1000000
      output_per_mtok_usd_micros: 0
      cached_per_mtok_usd_micros: 0
`)
	c, up := newTestClient(t, Config{Pricing: reg})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1000,"output_tokens":0}}`)

	tgt := target(up.URL)
	tgt.LogicalModelID = "m"
	resp, err := c.Generate(context.Background(), domain.VendorRequest{Target: tgt, Prompt: "hi"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Usage.CostMicros != 1000 {
		t.Errorf("cost = %d, want 1000", resp.Usage.CostMicros)
	}
}

func TestGenerate_WithoutARegistryReportsZeroCost(t *testing.T) {
	// A gateway running with no registry still dispatches; the call is recorded
	// at zero cost rather than failing or being dropped.
	c, up := newTestClient(t, Config{})
	up.respondJSON(`{"content":[{"type":"text","text":"ok"}],"usage":{"input_tokens":1000,"output_tokens":500}}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Usage.CostMicros != 0 {
		t.Errorf("cost = %d, want 0 with no registry", resp.Usage.CostMicros)
	}
	if resp.Usage.InputTokens != 1000 {
		t.Errorf("tokens were lost when the price table is absent: %+v", resp.Usage)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func TestIntPtr(t *testing.T) {
	// A non-positive ceiling must never reach the wire: max_tokens is required
	// and must be positive, so zero and a nonsense negative both collapse to the
	// conservative default rather than to a 400.
	cases := []struct{ in, want int }{
		{256, 256},
		{0, defaultMaxTokens},
		{-1, defaultMaxTokens},
		{defaultMaxTokens, defaultMaxTokens},
	}
	for _, tc := range cases {
		got := intPtr(tc.in)
		if got == nil {
			t.Fatalf("intPtr(%d) returned nil; max_tokens would be omitted", tc.in)
		}
		if *got != tc.want {
			t.Errorf("intPtr(%d) = %d, want %d", tc.in, *got, tc.want)
		}
	}
}

func TestFirstNonEmpty(t *testing.T) {
	if got := firstNonEmpty("", "b", "c"); got != "b" {
		t.Errorf("firstNonEmpty = %q", got)
	}
	if got := firstNonEmpty("", ""); got != "" {
		t.Errorf("firstNonEmpty = %q", got)
	}
}

func TestTruncate(t *testing.T) {
	if got := truncate("short", 10); got != "short" {
		t.Errorf("truncate = %q", got)
	}
	if got := truncate(strings.Repeat("x", 50), 10); len([]rune(got)) != 11 {
		t.Errorf("truncate did not bound the length: %d runes", len([]rune(got)))
	}
}

func TestNumberFrom(t *testing.T) {
	for _, v := range []any{1, int32(1), int64(1), float32(1), float64(1)} {
		if got, ok := numberFrom(v); !ok || got != 1 {
			t.Errorf("numberFrom(%T) = %v, %v", v, got, ok)
		}
	}
	// A string is not a number: silently coercing "0.5" would ship a knob the
	// caller never asked for.
	for _, v := range []any{"nope", true, nil, []any{1}} {
		if _, ok := numberFrom(v); ok {
			t.Errorf("numberFrom(%T) accepted a non-numeric value", v)
		}
	}
}

func mustMarshal(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "<unmarshalable>"
	}
	return string(b)
}

// TestCostMicros_CacheWritesArePricedSeparately guards a real under-billing.
// Anthropic reports cache-creation tokens as a SUBSET of input_tokens and
// charges a premium for them. Billing them at the plain input rate makes every
// prompt-caching workload look cheaper than it is.
func TestCostMicros_CacheWritesArePricedSeparately(t *testing.T) {
	c, _ := New(Config{Pricing: writeRegistryFile(t, `
models:
  - id: m
    provider: anthropic
    base_url: https://api.anthropic.com
    pricing:
      input_per_mtok_usd_micros: 1000000
      output_per_mtok_usd_micros: 0
      cached_per_mtok_usd_micros: 100000
      cache_write_per_mtok_usd_micros: 1250000
`)})

	// 400 plain + 600 cache-write, priced at 1.0x and 1.25x respectively.
	got := c.costMicros("m", 400, 0, 0, 600)
	want := int64(400) + int64(750)
	if got != want {
		t.Errorf("cost = %d, want %d (cache writes at the premium rate)", got, want)
	}
}

// TestToDomain_SplitsCacheWritesOutOfInput is the end-to-end version: the
// response parser must subtract cache writes from the plain input count before
// pricing, or they get billed at both rates.
func TestToDomain_SplitsCacheWritesOutOfInput(t *testing.T) {
	c, _ := New(Config{Pricing: writeRegistryFile(t, `
models:
  - id: m
    provider: anthropic
    base_url: https://api.anthropic.com
    pricing:
      input_per_mtok_usd_micros: 1000000
      output_per_mtok_usd_micros: 0
      cached_per_mtok_usd_micros: 100000
      cache_write_per_mtok_usd_micros: 1250000
`)})

	resp := c.toDomain(domain.VendorRequest{Target: domain.TargetModel{LogicalModelID: "m"}},
		messagesResponse{
			Content: []contentBlock{{Type: "text", Text: "ok"}},
			Usage: struct {
				InputTokens              int64 `json:"input_tokens"`
				OutputTokens             int64 `json:"output_tokens"`
				CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
				CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
			}{InputTokens: 1000, CacheCreationInputTokens: 600},
		})

	// 400 plain at 1.0x + 600 cache writes at 1.25x.
	if want := int64(400) + int64(750); resp.Usage.CostMicros != want {
		t.Errorf("cost = %d, want %d", resp.Usage.CostMicros, want)
	}
	// The reported token counts stay whole; only the pricing splits them.
	if resp.Usage.InputTokens != 1000 {
		t.Errorf("input tokens = %d, want the vendor's total 1000", resp.Usage.InputTokens)
	}
}
