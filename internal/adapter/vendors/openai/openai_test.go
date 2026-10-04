package openai

import (
	"context"
	"encoding/base64"
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

func newTestClient(t *testing.T, registryBody string) (*Client, *stubUpstream) {
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

	var reg *config.Registry
	if registryBody != "" {
		var err error
		reg, err = config.LoadRegistry(writeRegistryFile(t, registryBody))
		if err != nil {
			t.Fatalf("registry: %v", err)
		}
	}
	c, err := New(Config{HTTPClient: up.Client(), Pricing: reg})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return c, up
}

func writeRegistryFile(t *testing.T, body string) string {
	t.Helper()
	path := t.TempDir() + "/models.yaml"
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write registry: %v", err)
	}
	return path
}

func target(base string) domain.TargetModel {
	return domain.TargetModel{
		Vendor:              domain.VendorFamilyOpenAI,
		LogicalModelID:      "m",
		UpstreamModel:       "upstream-model",
		BaseURL:             base,
		ChatCompletionsPath: "/chat/completions",
		ImagesPath:          "/images/generations",
		EmbeddingsPath:      "/embeddings",
		MaxOutputTokens:     256,
		Capabilities:        []string{domain.CapabilityChat, domain.CapabilityImage, domain.CapabilityEmbeddings},
	}
}

func TestFamily(t *testing.T) {
	c, _ := newTestClient(t, "")
	if c.Family() != domain.VendorFamilyOpenAI {
		t.Errorf("Family = %q", c.Family())
	}
}

// ---------------------------------------------------------------------------
// Chat completions
// ---------------------------------------------------------------------------

func TestGenerate_ChatHappyPath(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{
		"model":"upstream-model",
		"choices":[{"message":{"role":"assistant","content":"the answer"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":12,"completion_tokens":4,
		         "prompt_tokens_details":{"cached_tokens":3}}
	}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:     target(up.URL),
		Prompt:     "hi",
		Credential: "sk-test",
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
	if resp.Usage.CachedTokens != 3 {
		t.Errorf("cached tokens = %d, want 3", resp.Usage.CachedTokens)
	}
	if resp.FinishReason != domain.FinishReasonComplete {
		t.Errorf("finish = %v", resp.FinishReason)
	}

	got := up.last(t)
	if got.Path != "/chat/completions" {
		t.Errorf("path = %q, want /chat/completions", got.Path)
	}
	if got.Method != http.MethodPost {
		t.Errorf("method = %q", got.Method)
	}
	// Auth is a bearer header, present because a credential was resolved.
	if auth := got.Header.Get("Authorization"); auth != "Bearer sk-test" {
		t.Errorf("authorization = %q", auth)
	}
	if got.Body["model"] != "upstream-model" {
		t.Errorf("upstream model sent = %v", got.Body["model"])
	}
}

func TestGenerate_NoCredentialSendsNoAuthHeader(t *testing.T) {
	// A local vLLM / Ollama server needs no key, and sending an empty bearer
	// makes some servers reject the request outright.
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"ok"}}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if _, present := up.last(t).Header["Authorization"]; present {
		t.Error("an Authorization header was sent with no credential")
	}
}

func TestGenerate_ForwardsTraceHeadersAndExtras(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"ok"}}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:      target(up.URL),
		Prompt:      "hi",
		Traceparent: "00-abc-def-01",
		Tracestate:  "vendor=1",
		ExtraHeaders: map[string]string{
			"HTTP-Referer": "https://openrouter.ai",
			"X-Title":      "chora",
		},
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	got := up.last(t)
	if got.Header.Get("traceparent") != "00-abc-def-01" {
		t.Errorf("traceparent = %q", got.Header.Get("traceparent"))
	}
	if got.Header.Get("tracestate") != "vendor=1" {
		t.Errorf("tracestate = %q", got.Header.Get("tracestate"))
	}
	// Provider-specific routing hints come from the registry, not the caller.
	if got.Header.Get("HTTP-Referer") != "https://openrouter.ai" {
		t.Errorf("referer = %q", got.Header.Get("HTTP-Referer"))
	}
}

func TestGenerate_SendsSystemPromptFirst(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"ok"}}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:       target(up.URL),
		Prompt:       "hi",
		SystemPrompt: "be terse",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	msgs, _ := up.last(t).Body["messages"].([]any)
	if len(msgs) != 2 {
		t.Fatalf("messages = %v, want system + user", up.last(t).Body["messages"])
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "system" || first["content"] != "be terse" {
		t.Errorf("first message = %v", first)
	}
}

func TestGenerate_ContentsJSONWinsOverPrompt(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"ok"}}]}`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:       target(up.URL),
		Prompt:       "ignored",
		ContentsJSON: `[{"role":"user","content":"first"},{"role":"assistant","content":"second"},{"role":"user","content":"third"}]`,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	msgs, _ := up.last(t).Body["messages"].([]any)
	if len(msgs) != 3 {
		t.Errorf("messages = %d, want the 3-turn conversation", len(msgs))
	}
}

func TestGenerate_MalformedContentsFallsBackToPrompt(t *testing.T) {
	// A malformed conversation is a caller bug, but dropping the whole call
	// would be worse than sending the bare prompt.
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"ok"}}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "the real prompt", ContentsJSON: `{not json`,
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(mustMarshal(up.last(t).Body), "the real prompt") {
		t.Errorf("fallback did not carry the prompt: %s", mustMarshal(up.last(t).Body))
	}
}

func TestGenerate_ForwardsGenerationConfig(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"ok"}}]}`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL),
		Prompt: "hi",
		GenerationConfig: map[string]any{
			"temperature":           0.3,
			"top_p":                 0.8,
			"max_tokens":            float64(64),
			"max_completion_tokens": float64(64),
			"stop":                  []string{"END", "HALT"},
			"seed":                  float64(9),
			"n":                     float64(1),
		},
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	body := up.last(t).Body
	if body["temperature"] != 0.3 || body["top_p"] != 0.8 || body["max_tokens"] != float64(64) {
		t.Errorf("core knobs = %s", mustMarshal(body))
	}
	if body["seed"] != float64(9) {
		t.Errorf("seed = %v", body["seed"])
	}
	if body["n"] != float64(1) {
		t.Errorf("n = %v", body["n"])
	}
	stops, _ := body["stop"].([]any)
	if len(stops) != 2 {
		t.Errorf("stop = %v, want both sequences", body["stop"])
	}
}

func TestGenerate_StreamIsNeverRequested(t *testing.T) {
	// Sending stream:true and buffering the SSE stream as a string would look
	// like success to the caller while delivering a broken body.
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"ok"}}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "hi",
		GenerationConfig: map[string]any{"stream": true},
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if v, present := up.last(t).Body["stream"]; present && v != false {
		t.Errorf("stream = %v, want it never sent as true", v)
	}
}

func TestGenerate_ForwardsTools(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"ok"}}]}`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:    target(up.URL),
		Prompt:    "hi",
		ToolsJSON: `[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}]`,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(mustMarshal(up.last(t).Body), "lookup") {
		t.Errorf("tools dropped: %s", mustMarshal(up.last(t).Body))
	}
}

func TestGenerate_MalformedToolsIsDroppedNotFatal(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"ok"}}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi", ToolsJSON: `{broken`,
	}); err != nil {
		t.Fatalf("a malformed tools blob should not fail the call: %v", err)
	}
}

func TestGenerate_ToolCallsAreReturnedAsJSON(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"role":"assistant","content":"",
		"tool_calls":[{"id":"call_1","type":"function",
		  "function":{"name":"lookup","arguments":"{\"q\":\"x\"}"}}]},
		"finish_reason":"tool_calls"}]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:    target(up.URL),
		Prompt:    "hi",
		ToolsJSON: `[{"type":"function","function":{"name":"lookup"}}]`,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if !strings.Contains(resp.ToolCallsJSON, "lookup") {
		t.Errorf("tool calls = %s", resp.ToolCallsJSON)
	}
	// A tool-calling turn is a successful turn, not a truncated one.
	if resp.FinishReason != domain.FinishReasonComplete {
		t.Errorf("finish = %v, want complete", resp.FinishReason)
	}
}

func TestGenerate_MultimodalContentIsFlattened(t *testing.T) {
	// Vision-capable servers return content as an array of typed parts.
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"role":"assistant","content":[
		{"type":"text","text":"part one "},{"type":"text","text":"part two"}]}}]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{Target: target(up.URL), Prompt: "hi"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.Completion != "part one part two" {
		t.Errorf("completion = %q, want the parts joined", resp.Completion)
	}
}

func TestGenerate_ZeroChoicesIsReported(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{Target: target(up.URL), Prompt: "hi"})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.FinishReason != domain.FinishReasonUnspecified {
		t.Errorf("finish = %v, want unspecified with a detail", resp.FinishReason)
	}
	if resp.FinishDetail == "" {
		t.Error("want a detail explaining the empty choices")
	}
}

func TestGenerate_UpstreamErrorCarriesStatus(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"invalid api key"}}`))
	})

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "hi", Credential: "sk-bad",
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	up2, ok := err.(*UpstreamError)
	if !ok {
		t.Fatalf("error = %T, want *UpstreamError", err)
	}
	if up2.UpstreamStatusCode() != http.StatusUnauthorized {
		t.Errorf("status = %d", up2.UpstreamStatusCode())
	}
	if !strings.Contains(up2.Body, "invalid api key") {
		t.Errorf("body = %q, want the provider's message", up2.Body)
	}
}

func TestGenerate_MalformedUpstreamJSON(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`<html>not json</html>`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{Target: target(up.URL), Prompt: "hi"}); err == nil {
		t.Fatal("expected a decode error")
	}
}

func TestGenerate_UnreachableUpstream(t *testing.T) {
	c, up := newTestClient(t, "")
	up.Close()

	if _, err := c.Generate(context.Background(), domain.VendorRequest{Target: target(up.URL), Prompt: "hi"}); err == nil {
		t.Fatal("expected a transport error")
	}
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

func TestGenerate_ImageFromBase64(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[{"b64_json":"` +
		base64.StdEncoding.EncodeToString([]byte("PNGDATA")) +
		`","revised_prompt":"a better cat"}]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "a cat",
		ResponseModality: domain.ModalityImage,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if string(resp.ImageBytes) != "PNGDATA" {
		t.Errorf("image bytes = %q", resp.ImageBytes)
	}
	if resp.ImageMIMEType != "image/png" {
		t.Errorf("mime = %q", resp.ImageMIMEType)
	}
	if resp.ImageRevisedPrompt != "a better cat" {
		t.Errorf("revised prompt = %q", resp.ImageRevisedPrompt)
	}
	if got := up.last(t); got.Path != "/images/generations" {
		t.Errorf("path = %q, want /images/generations", got.Path)
	}
}

func TestGenerate_ImageForwardsOnlyImageKnobs(t *testing.T) {
	// Sending temperature to an images endpoint is a 400 on OpenAI.
	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[{"b64_json":"` + base64.StdEncoding.EncodeToString([]byte("X")) + `"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "a cat",
		ResponseModality: domain.ModalityImage,
		GenerationConfig: map[string]any{
			"size": "1024x1024", "quality": "high", "style": "vivid",
			"temperature": 0.9, // must NOT be forwarded
		},
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	body := up.last(t).Body
	if body["size"] != "1024x1024" || body["quality"] != "high" || body["style"] != "vivid" {
		t.Errorf("image knobs = %s", mustMarshal(body))
	}
	if _, present := body["temperature"]; present {
		t.Errorf("temperature leaked to the images surface: %s", mustMarshal(body))
	}
	if body["prompt"] != "a cat" {
		t.Errorf("prompt = %v", body["prompt"])
	}
}

func TestGenerate_ImageFromURLIsFetched(t *testing.T) {
	// A provider that returns a URL rather than bytes must still yield bytes:
	// the facade's contract is bytes, and mixing shapes per provider would be
	// worse for the caller than always returning bytes.
	img := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "image/webp")
		_, _ = w.Write([]byte("WEBPBYTES"))
	}))
	t.Cleanup(img.Close)

	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[{"url":"` + img.URL + `"}]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "a cat",
		ResponseModality: domain.ModalityImage,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if string(resp.ImageBytes) != "WEBPBYTES" {
		t.Errorf("bytes = %q", resp.ImageBytes)
	}
	if resp.ImageMIMEType != "image/webp" {
		t.Errorf("mime = %q, want it taken from the download response", resp.ImageMIMEType)
	}
}

func TestGenerate_ImageFetchDoesNotLeakTheCredential(t *testing.T) {
	// The download URL is provider-minted and may point at a third-party host;
	// attaching the API key there would hand it over.
	img := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, present := r.Header["Authorization"]; present {
			t.Error("the credential was forwarded to the image host")
		}
		_, _ = w.Write([]byte("X"))
	}))
	t.Cleanup(img.Close)

	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[{"url":"` + img.URL + `"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           target(up.URL),
		Prompt:           "cat",
		ResponseModality: domain.ModalityImage,
		Credential:       "sk-secret",
	}); err != nil {
		t.Fatalf("Generate: %v", err)
	}
}

func TestGenerate_ImageFetchFailure(t *testing.T) {
	img := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	t.Cleanup(img.Close)

	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[{"url":"` + img.URL + `"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "cat", ResponseModality: domain.ModalityImage,
	}); err == nil {
		t.Fatal("expected an error when the image download 404s")
	}
}

func TestGenerate_ImageMalformedBase64FailsLoud(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[{"b64_json":"!!!not-base64!!!"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "cat", ResponseModality: domain.ModalityImage,
	}); err == nil {
		t.Fatal("expected an error; a zero-byte image is a silent failure")
	}
}

func TestGenerate_ImageRequiresPrompt(t *testing.T) {
	c, _ := newTestClient(t, "")
	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target("http://x"), ResponseModality: domain.ModalityImage,
	}); err == nil {
		t.Fatal("expected an error for an image request with no prompt")
	}
}

func TestGenerate_ImageWithoutData(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "cat", ResponseModality: domain.ModalityImage,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if resp.FinishDetail == "" {
		t.Error("want a detail explaining the empty data array")
	}
}

func TestGenerate_ImageWithNeitherBytesNorURL(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[{"revised_prompt":"only this"}]}`)

	if _, err := c.Generate(context.Background(), domain.VendorRequest{
		Target: target(up.URL), Prompt: "cat", ResponseModality: domain.ModalityImage,
	}); err == nil {
		t.Fatal("expected an error when the provider returns neither bytes nor a url")
	}
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

func TestEmbedText(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"model":"emb","data":[{"embedding":[0.1,0.2]}],"usage":{"prompt_tokens":6}}`)

	resp, err := c.EmbedText(context.Background(), domain.EmbedVendorRequest{
		Target: target(up.URL), Text: "hello", Credential: "sk-x",
	})
	if err != nil {
		t.Fatalf("EmbedText: %v", err)
	}
	if len(resp.Values) != 2 {
		t.Errorf("values = %v", resp.Values)
	}
	if resp.InputTokens != 6 {
		t.Errorf("input tokens = %d", resp.InputTokens)
	}
	if resp.ModelVersion != "emb" {
		t.Errorf("model = %q", resp.ModelVersion)
	}
	got := up.last(t)
	if got.Path != "/embeddings" {
		t.Errorf("path = %q", got.Path)
	}
	if got.Body["input"] != "hello" {
		t.Errorf("input = %v", got.Body["input"])
	}
}

func TestEmbedText_ForwardsDimensions(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[{"embedding":[0.1]}]}`)

	if _, err := c.EmbedText(context.Background(), domain.EmbedVendorRequest{
		Target: target(up.URL), Text: "hello", OutputDimensions: 256,
	}); err != nil {
		t.Fatalf("EmbedText: %v", err)
	}
	if up.last(t).Body["dimensions"] != float64(256) {
		t.Errorf("dimensions = %v", up.last(t).Body["dimensions"])
	}
}

func TestEmbedText_FallsBackToLogicalModelName(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[{"embedding":[0.1]}]}`)

	tgt := target(up.URL)
	tgt.UpstreamModel = "" // an entry that never set upstream_model
	if _, err := c.EmbedText(context.Background(), domain.EmbedVendorRequest{
		Target: tgt, LogicalModelID: "named-model", Text: "hello",
	}); err != nil {
		t.Fatalf("EmbedText: %v", err)
	}
	if up.last(t).Body["model"] != "named-model" {
		t.Errorf("model = %v, want the logical id used", up.last(t).Body["model"])
	}
}

func TestEmbedText_EmptyDataFailsLoud(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"data":[]}`)

	if _, err := c.EmbedText(context.Background(), domain.EmbedVendorRequest{
		Target: target(up.URL), Text: "hello",
	}); err == nil {
		t.Fatal("expected an error; a silently empty vector corrupts every store consuming it")
	}
}

// ---------------------------------------------------------------------------
// Cost
// ---------------------------------------------------------------------------

func TestCostMicros(t *testing.T) {
	reg, err := config.LoadRegistry(writeRegistryFile(t, `
models:
  - id: priced
    provider: openai
    base_url: https://x.test/v1
    pricing:
      input_per_mtok_usd_micros: 1000000
      output_per_mtok_usd_micros: 2000000
      cached_per_mtok_usd_micros: 100000
  - id: free
    provider: openai
    base_url: https://x.test/v1
`))
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	c, err := New(Config{Pricing: reg})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// 1000 in (500 of them cached) + 2000 out at $1 / $2 / $0.10 per Mtok.
	// The cached half is billed at the cache rate, NOT at the input rate, so
	// the billable input is 500 not 1000.
	got := c.costMicros("priced", 1000, 2000, 500)
	want := int64(500) + int64(4000) + int64(50)
	if got != want {
		t.Errorf("cost = %d, want %d", got, want)
	}

	// An unpriced model costs zero: the honest answer for a self-hosted or
	// operator-supplied model.
	if v := c.costMicros("free", 1000, 2000, 0); v != 0 {
		t.Errorf("unpriced cost = %d, want 0", v)
	}
	if v := c.costMicros("ghost", 1000, 2000, 0); v != 0 {
		t.Errorf("unknown model cost = %d, want 0", v)
	}

	// No registry at all is zero rather than a panic.
	bare, err := New(Config{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if v := bare.costMicros("priced", 1000, 2000, 0); v != 0 {
		t.Errorf("nil-registry cost = %d, want 0", v)
	}
}

func TestCostMicros_CachedTokensAreNotDoubleCounted(t *testing.T) {
	// Cached input is a subset of input; billing it at both rates would
	// overcharge every cached request.
	reg, err := config.LoadRegistry(writeRegistryFile(t, `
models:
  - id: m
    provider: openai
    base_url: https://x.test/v1
    pricing:
      input_per_mtok_usd_micros: 1000000
      output_per_mtok_usd_micros: 0
      cached_per_mtok_usd_micros: 0
`))
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	c, _ := New(Config{Pricing: reg})
	if v := c.costMicros("m", 1000, 0, 1000); v != 0 {
		t.Errorf("cost = %d, want 0 (the whole input was cached)", v)
	}
	// A nonsensical cached > input must not produce a negative debit.
	if v := c.costMicros("m", 100, 0, 500); v != 0 {
		t.Errorf("cost = %d, want 0 rather than a negative bill", v)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func TestContentToString(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want string
	}{
		{"nil", nil, ""},
		{"string", "plain", "plain"},
		{"array", []any{map[string]any{"type": "text", "text": "a"}, map[string]any{"text": "b"}}, "ab"},
		{"unknown", 42, ""},
	}
	for _, tc := range cases {
		if got := contentToString(tc.in); got != tc.want {
			t.Errorf("%s: contentToString = %q, want %q", tc.name, got, tc.want)
		}
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

func TestFirstNonEmpty(t *testing.T) {
	if got := firstNonEmpty("", "b", "c"); got != "b" {
		t.Errorf("firstNonEmpty = %q", got)
	}
	if got := firstNonEmpty("", ""); got != "" {
		t.Errorf("firstNonEmpty = %q", got)
	}
}

func TestNumberFrom(t *testing.T) {
	for _, v := range []any{1, int32(1), int64(1), float32(1), float64(1)} {
		if _, ok := numberFrom(v); !ok {
			t.Errorf("numberFrom(%T) failed", v)
		}
	}
	if _, ok := numberFrom("nope"); ok {
		t.Error("numberFrom accepted a string")
	}
}

func mustMarshal(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return "<unmarshalable>"
	}
	return string(b)
}
