package openai

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

func groundedTarget(base string, g *domain.Grounding) domain.TargetModel {
	t := target(base)
	t.Grounding = g
	return t
}

func mustBody(t *testing.T, m map[string]any) string {
	t.Helper()
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

// ---------------------------------------------------------------------------
// Request shape
// ---------------------------------------------------------------------------

func TestBuildResponsesBody_SearchToolAndInput(t *testing.T) {
	g := &domain.Grounding{}
	body := buildResponsesBody(domain.VendorRequest{
		Target:     groundedTarget("https://x.test/v1", g),
		Prompt:     "Who won the most recent Formula 1 race?",
		Credential: "sk-test",
	})

	if body["model"] != "upstream-model" {
		t.Errorf("model = %v", body["model"])
	}
	// A single prompt collapses to the scalar `input` form, which is what the
	// docs show and what every implementation definitely parses.
	if body["input"] != "Who won the most recent Formula 1 race?" {
		t.Errorf("input = %v", body["input"])
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("tools = %s", mustBody(t, body))
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "web_search" {
		t.Errorf("tool type = %v, want web_search (the default for an openai provider)", tool["type"])
	}
	// Streaming is never requested on this surface either.
	if body["stream"] != false {
		t.Errorf("stream = %v, want false", body["stream"])
	}
}

func TestBuildResponsesBody_SystemPromptUsesInstructions(t *testing.T) {
	body := buildResponsesBody(domain.VendorRequest{
		Target:       groundedTarget("https://x.test/v1", &domain.Grounding{}),
		Prompt:       "hi",
		SystemPrompt: "be terse",
	})
	// The Responses API carries the system prompt in `instructions`, not as a
	// system message in the input array.
	if body["instructions"] != "be terse" {
		t.Errorf("instructions = %v", body["instructions"])
	}
}

func TestBuildResponsesBody_ConversationBecomesAnItemArray(t *testing.T) {
	body := buildResponsesBody(domain.VendorRequest{
		Target: groundedTarget("https://x.test/v1", &domain.Grounding{}),
		Prompt: "ignored",
		ContentsJSON: `[{"role":"user","content":"first"},{"role":"assistant","content":"second"},` +
			`{"role":"user","content":"third"}]`,
	})
	raw, ok := body["input"].(json.RawMessage)
	if !ok {
		t.Fatalf("input = %T, want a raw item array for a multi-turn conversation", body["input"])
	}
	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil {
		t.Fatalf("input items are not parseable: %v", err)
	}
	if len(items) != 3 {
		t.Errorf("items = %d, want 3", len(items))
	}
}

func TestBuildResponsesBody_MalformedConversationFallsBackToPrompt(t *testing.T) {
	body := buildResponsesBody(domain.VendorRequest{
		Target:       groundedTarget("https://x.test/v1", &domain.Grounding{}),
		Prompt:       "the real prompt",
		ContentsJSON: `{broken`,
	})
	if body["input"] != "the real prompt" {
		t.Errorf("input = %v, want the prompt fallback", body["input"])
	}
}

func TestBuildResponsesBody_ForwardsGenerationConfig(t *testing.T) {
	body := buildResponsesBody(domain.VendorRequest{
		Target: groundedTarget("https://x.test/v1", &domain.Grounding{}),
		Prompt: "hi",
		GenerationConfig: map[string]any{
			"max_output_tokens": float64(2048),
			"reasoning_effort":  "high",
			"temperature":       0.4,
			// Not a Responses field; must not be forwarded.
			"top_k": float64(40),
		},
	})
	if body["max_output_tokens"] != float64(2048) {
		t.Errorf("max_output_tokens = %v", body["max_output_tokens"])
	}
	if body["reasoning_effort"] != "high" {
		t.Errorf("reasoning_effort = %v", body["reasoning_effort"])
	}
	if body["temperature"] != 0.4 {
		t.Errorf("temperature = %v", body["temperature"])
	}
	if _, present := body["top_k"]; present {
		t.Errorf("top_k leaked onto the Responses surface: %s", mustBody(t, body))
	}
}

// ---------------------------------------------------------------------------
// Tool shape per surface
// ---------------------------------------------------------------------------

func TestBuildGroundingTools(t *testing.T) {
	cases := []struct {
		name      string
		grounding *domain.Grounding
		vendor    domain.VendorFamily
		wantType  string
		wantName  string
		wantMax   any
	}{
		{
			name:      "openai defaults to the Responses tool",
			grounding: &domain.Grounding{},
			vendor:    domain.VendorFamilyOpenAI,
			wantType:  "web_search",
		},
		{
			name:      "anthropic defaults to its dated server tool and needs a name",
			grounding: &domain.Grounding{},
			vendor:    domain.VendorFamilyAnthropic,
			wantType:  "web_search_20250305",
			wantName:  "web_search",
		},
		{
			name:      "chat_completions defaults to the preview tool",
			grounding: &domain.Grounding{Surface: domain.SurfaceChatCompletions},
			vendor:    domain.VendorFamilyOpenAI,
			wantType:  "web_search_preview",
		},
		{
			name:      "an explicit tool_type wins, for a provider that renamed it",
			grounding: &domain.Grounding{ToolType: "web_search_2025_08_26"},
			vendor:    domain.VendorFamilyOpenAI,
			wantType:  "web_search_2025_08_26",
		},
		{
			name:      "max_uses is forwarded when set",
			grounding: &domain.Grounding{MaxUses: 5},
			vendor:    domain.VendorFamilyAnthropic,
			wantType:  "web_search_20250305",
			wantName:  "web_search",
			wantMax:   5,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tgt := domain.TargetModel{Vendor: tc.vendor, Grounding: tc.grounding}
			tools := buildGroundingTools(domain.VendorRequest{Target: tgt})
			if len(tools) != 1 {
				t.Fatalf("tools = %v, want exactly one", tools)
			}
			tool, _ := tools[0].(map[string]any)
			if tool["type"] != tc.wantType {
				t.Errorf("type = %v, want %q", tool["type"], tc.wantType)
			}
			if tc.wantName != "" && tool["name"] != tc.wantName {
				t.Errorf("name = %v, want %q", tool["name"], tc.wantName)
			}
			if tc.wantMax != nil && tool["max_uses"] != tc.wantMax {
				t.Errorf("max_uses = %v, want %v", tool["max_uses"], tc.wantMax)
			}
			if tc.wantMax == nil {
				if _, present := tool["max_uses"]; present {
					t.Errorf("max_uses was set with no configured value: %v", tool["max_uses"])
				}
			}
		})
	}
}

func TestBuildGroundingTools_ExtraFieldsMergeButCannotOverrideType(t *testing.T) {
	g := &domain.Grounding{
		MaxUses: 3,
		ExtraToolFields: map[string]any{
			"search_context_size": "medium",
			// A stray `type` in extra_tool_fields must not break the tool.
			"type": "something-else",
		},
	}
	tools := buildGroundingTools(domain.VendorRequest{
		Target: domain.TargetModel{Vendor: domain.VendorFamilyOpenAI, Grounding: g},
	})
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "web_search" {
		t.Errorf("type = %v, want the computed discriminator to win", tool["type"])
	}
	if tool["search_context_size"] != "medium" {
		t.Errorf("extra field dropped: %v", tool)
	}
	if tool["max_uses"] != 3 {
		t.Errorf("max_uses = %v, want 3", tool["max_uses"])
	}
}

func TestBuildGroundingTools_NilGrounding(t *testing.T) {
	if tools := buildGroundingTools(domain.VendorRequest{Target: domain.TargetModel{}}); tools != nil {
		t.Errorf("tools = %v, want nil for a target with no grounding configured", tools)
	}
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

// TestResponsesToDomain_ParsesOutputByType is the test that matters most.
//
// A reasoning model interleaves `reasoning`, `web_search_call` and `message`
// items in the output array, so indexing by position — which is the obvious
// implementation — returns a reasoning blob or nothing at all.
func TestResponsesToDomain_ParsesOutputByType(t *testing.T) {
	c, _ := newTestClient(t, "")
	req := domain.VendorRequest{
		Target: groundedTarget("https://x.test/v1", &domain.Grounding{}),
		Prompt: "who won",
	}

	resp := c.responsesToDomain(req, responsesReply{
		Model: "gpt-5",
		Output: []responsesItem{
			{Type: "reasoning", ID: "rs_1"},
			{Type: "web_search_call", ID: "ws_1", Status: "completed",
				Action: struct {
					Type   string   `json:"type"`
					Query  string   `json:"query"`
					Querys []string `json:"queries"`
				}{Type: "search", Query: "most recent Formula 1 race winner"}},
			{Type: "message", Role: "assistant", Content: []responsesContent{{
				Type: "output_text",
				Text: "Max Verstappen won.",
				Annotations: []responsesAnnotation{
					{Type: "url_citation", URL: "https://f1.example/race", Title: "Race report", StartIndex: intPtr(0), EndIndex: intPtr(3)},
				},
			}}},
		},
		Usage: struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
			TotalTokens  int64 `json:"total_tokens"`
			InputDetails struct {
				CachedTokens int64 `json:"cached_tokens"`
			} `json:"input_tokens_details"`
			OutputDetails struct {
				ReasoningTokens int64 `json:"reasoning_tokens"`
			} `json:"output_tokens_details"`
		}{InputTokens: 20, OutputTokens: 5},
	})

	if resp.Completion != "Max Verstappen won." {
		t.Errorf("completion = %q; the message item was not found by type", resp.Completion)
	}
	if len(resp.Citations) != 1 {
		t.Fatalf("citations = %+v, want 1", resp.Citations)
	}
	if resp.Citations[0].URL != "https://f1.example/race" {
		t.Errorf("citation url = %q", resp.Citations[0].URL)
	}
	if resp.Citations[0].Title != "Race report" {
		t.Errorf("citation title = %q", resp.Citations[0].Title)
	}
	if len(resp.SearchQueries) != 1 || resp.SearchQueries[0] != "most recent Formula 1 race winner" {
		t.Errorf("search queries = %v", resp.SearchQueries)
	}
}

func TestResponsesToDomain_MultipleSearchQueries(t *testing.T) {
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{Output: []responsesItem{
			{Type: "web_search_call", Action: struct {
				Type   string   `json:"type"`
				Query  string   `json:"query"`
				Querys []string `json:"queries"`
			}{Type: "search", Query: "first", Querys: []string{"second", "third"}}},
		}},
	)
	if len(resp.SearchQueries) != 3 {
		t.Errorf("queries = %v, want all three (the singular field plus the plural)", resp.SearchQueries)
	}
	if resp.SearchQueries[0] != "first" {
		t.Errorf("query order = %v", resp.SearchQueries)
	}
}

func TestResponsesToDomain_SkipsNonURLCitations(t *testing.T) {
	// The annotations array can also carry file_citation entries when
	// file_search is in play. Those are not web sources and must not be
	// presented as if they were.
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{Output: []responsesItem{
			{Type: "message", Content: []responsesContent{{
				Type: "output_text",
				Text: "answer",
				Annotations: []responsesAnnotation{
					{Type: "file_citation", URL: "https://files.example/a.pdf"},
					{Type: "url_citation", URL: "https://web.example/a"},
					{Type: "url_citation"}, // no url at all
				},
			}}},
		}},
	)
	if len(resp.Citations) != 1 {
		t.Fatalf("citations = %+v, want only the one real web source", resp.Citations)
	}
	if resp.Citations[0].URL != "https://web.example/a" {
		t.Errorf("kept the wrong citation: %+v", resp.Citations[0])
	}
}

func TestResponsesToDomain_ConcatenatesMultipleTextParts(t *testing.T) {
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{Output: []responsesItem{
			{Type: "message", Content: []responsesContent{
				{Type: "output_text", Text: "part one. "},
				{Type: "output_text", Text: "part two."},
			}},
		}},
	)
	if resp.Completion != "part one. part two." {
		t.Errorf("completion = %q, want both parts joined", resp.Completion)
	}
}

func TestResponsesToDomain_EmptyOutputIsReported(t *testing.T) {
	// A grounded call that returns nothing must say so, not look like a
	// successful empty answer.
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{},
	)
	if resp.FinishReason != domain.FinishReasonUnspecified {
		t.Errorf("finish = %v, want unspecified", resp.FinishReason)
	}
	if resp.FinishDetail == "" {
		t.Error("want a detail explaining the empty output")
	}
}

func TestResponsesToDomain_CostFromRegistry(t *testing.T) {
	c, _ := newTestClient(t, `
models:
  - id: m
    provider: openai
    base_url: https://x.test/v1
    pricing:
      input_per_mtok_usd_micros: 1000000
      output_per_mtok_usd_micros: 2000000
`)
	req := domain.VendorRequest{
		Target: domain.TargetModel{
			Vendor: domain.VendorFamilyOpenAI, LogicalModelID: "m",
			UpstreamModel: "m", BaseURL: "https://x.test/v1",
		},
	}
	resp := c.responsesToDomain(req, responsesReply{
		Output: []responsesItem{
			{Type: "message", Content: []responsesContent{{Type: "output_text", Text: "x"}}},
		},
		Usage: struct {
			InputTokens  int64 `json:"input_tokens"`
			OutputTokens int64 `json:"output_tokens"`
			TotalTokens  int64 `json:"total_tokens"`
			InputDetails struct {
				CachedTokens int64 `json:"cached_tokens"`
			} `json:"input_tokens_details"`
			OutputDetails struct {
				ReasoningTokens int64 `json:"reasoning_tokens"`
			} `json:"output_tokens_details"`
		}{InputTokens: 1000, OutputTokens: 2000},
	})
	// 1000 in at $1/Mtok + 2000 out at $2/Mtok.
	if want := int64(1000) + int64(4000); resp.Usage.CostMicros != want {
		t.Errorf("cost = %d, want %d", resp.Usage.CostMicros, want)
	}
}

// ---------------------------------------------------------------------------
// End-to-end through Generate
// ---------------------------------------------------------------------------

func TestGenerate_GroundedDispatch(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{
		"id":"resp_1","model":"upstream-model","status":"completed",
		"output":[
			{"type":"web_search_call","id":"ws_1","status":"completed",
			 "action":{"type":"search","query":"latest news"}},
			{"type":"message","role":"assistant","content":[{"type":"output_text",
			 "text":"Here is what happened.","annotations":[
			   {"type":"url_citation","url":"https://news.example/a","title":"News"}]}]}
		],
		"usage":{"input_tokens":15,"output_tokens":9,"total_tokens":24}
	}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           groundedTarget(up.URL, &domain.Grounding{}),
		Prompt:           "what happened?",
		ResponseModality: domain.ModalityGrounded,
		Credential:       "sk-test",
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	got := up.last(t)
	if got.Path != "/responses" {
		t.Errorf("path = %q, want /responses", got.Path)
	}
	if resp.Completion != "Here is what happened." {
		t.Errorf("completion = %q", resp.Completion)
	}
	if len(resp.Citations) != 1 {
		t.Errorf("citations = %+v", resp.Citations)
	}
	if resp.Usage.InputTokens != 15 || resp.Usage.OutputTokens != 9 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestGenerate_GroundedUsesTheConfiguredResponsesPath(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"output":[{"type":"message","content":[{"type":"output_text","text":"x"}]}]}`)

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           groundedTarget(up.URL, &domain.Grounding{ResponsesPath: "/custom/responses"}),
		Prompt:           "hi",
		ResponseModality: domain.ModalityGrounded,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if got := up.last(t); got.Path != "/custom/responses" {
		t.Errorf("path = %q, want the registry-configured one", got.Path)
	}
}

func TestGenerate_GroundedUpstreamError(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"search quota exceeded"}`))
	})

	_, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           groundedTarget(up.URL, &domain.Grounding{}),
		Prompt:           "hi",
		ResponseModality: domain.ModalityGrounded,
	})
	if err == nil {
		t.Fatal("expected an error")
	}
	ue, ok := err.(*UpstreamError)
	if !ok {
		t.Fatalf("error = %T, want *UpstreamError", err)
	}
	if ue.UpstreamStatusCode() != http.StatusTooManyRequests {
		t.Errorf("status = %d", ue.UpstreamStatusCode())
	}
}

// TestGenerate_ChatCompletionsGrounding proves the classic surface also carries
// a search tool when the registry says so.
func TestGenerate_ChatCompletionsGrounding(t *testing.T) {
	c, up := newTestClient(t, "")
	up.respondJSON(`{"choices":[{"message":{"content":"grounded answer",
		"annotations":[{"type":"url_citation","url":"https://web.example/x","title":"T"}]}}]}`)

	resp, err := c.Generate(context.Background(), domain.VendorRequest{
		Target:           groundedTarget(up.URL, &domain.Grounding{Surface: domain.SurfaceChatCompletions}),
		Prompt:           "hi",
		ResponseModality: domain.ModalityGrounded,
	})
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	got := up.last(t)
	if got.Path != "/chat/completions" {
		t.Errorf("path = %q, want chat/completions", got.Path)
	}
	body := mustBody(t, got.Body)
	if !contains(body, "web_search_preview") {
		t.Errorf("the search tool was not attached: %s", body)
	}
	if len(resp.Citations) != 1 {
		t.Errorf("chat-completions citations were dropped: %+v", resp.Citations)
	}
}

func contains(haystack, needle string) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		if haystack[i:i+len(needle)] == needle {
			return true
		}
	}
	return false
}

func intPtr(v int) *int { return &v }

// ---------------------------------------------------------------------------
// Terminal status
// ---------------------------------------------------------------------------

// TestResponsesToDomain_IncompleteIsNotSuccess guards the bug the live Meta run
// exposed: a research run that exhausts max_output_tokens returns
// `status: "incomplete"` carrying narration and NO answer. Reporting that as a
// successful completion hands the caller a confident non-answer.
func TestResponsesToDomain_IncompleteIsNotSuccess(t *testing.T) {
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{
			Status: "incomplete",
			IncompleteDetails: &struct {
				Reason string `json:"reason"`
			}{Reason: "max_output_tokens"},
			Output: []responsesItem{
				{Type: "reasoning"},
				{Type: "message", Content: []responsesContent{{Type: "output_text", Text: "I'll look that up."}}},
				{Type: "web_search_call", Action: struct {
					Type   string   `json:"type"`
					Query  string   `json:"query"`
					Querys []string `json:"queries"`
				}{Type: "search", Query: "population of Singapore"}},
			},
		},
	)

	if resp.FinishReason != domain.FinishReasonMaxTokens {
		t.Errorf("finish reason = %v, want max_tokens", resp.FinishReason)
	}
	if !contains(resp.FinishDetail, "max_output_tokens") {
		t.Errorf("detail = %q, want it to name the truncation cause", resp.FinishDetail)
	}
	// The partial text is still returned — it is the only evidence of what
	// happened, and what the caller needs in order to raise the ceiling.
	if resp.Completion != "I'll look that up." {
		t.Errorf("completion = %q, want the partial text preserved", resp.Completion)
	}
}

func TestResponsesToDomain_IncompleteWithoutDetailsStillExplains(t *testing.T) {
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{
			Status: "incomplete",
			Output: []responsesItem{
				{Type: "message", Content: []responsesContent{{Type: "output_text", Text: "partial"}}},
			},
		},
	)
	if resp.FinishReason != domain.FinishReasonMaxTokens {
		t.Errorf("finish reason = %v, want max_tokens", resp.FinishReason)
	}
	if resp.FinishDetail == "" {
		t.Error("want a detail even with no incomplete_details")
	}
}

func TestResponsesToDomain_FailedStatusIsAVendorError(t *testing.T) {
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{
			Status: "failed",
			Error: &struct {
				Code    string `json:"code"`
				Message string `json:"message"`
			}{Code: "server_error", Message: "upstream exploded"},
		},
	)
	if resp.FinishReason != domain.FinishReasonVendorError {
		t.Errorf("finish reason = %v, want vendor_error", resp.FinishReason)
	}
	if !contains(resp.FinishDetail, "upstream exploded") {
		t.Errorf("detail = %q, want the provider's message", resp.FinishDetail)
	}
}

func TestResponsesToDomain_UnknownStatusIsReported(t *testing.T) {
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{
			Status: "in_progress",
			Output: []responsesItem{
				{Type: "message", Content: []responsesContent{{Type: "output_text", Text: "partial"}}},
			},
		},
	)
	if resp.FinishReason == domain.FinishReasonComplete {
		t.Error("an in_progress reply was reported as complete")
	}
	if !contains(resp.FinishDetail, "in_progress") {
		t.Errorf("detail = %q, want it to name the status", resp.FinishDetail)
	}
}

func TestResponsesToDomain_CompletedWithEmptyOutput(t *testing.T) {
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{Status: "completed"},
	)
	if resp.FinishReason != domain.FinishReasonUnspecified {
		t.Errorf("finish reason = %v, want unspecified", resp.FinishReason)
	}
	if resp.FinishDetail == "" {
		t.Error("want a detail explaining the empty output")
	}
}

// TestSearchQueriesFrom_IgnoresOpenPage guards the second real finding: a search
// call's action is not always a search. A reasoning model also emits
// `{"type":"open_page","url":"..."}` once it reads a result, and reporting that
// URL as a search query would mislead whoever debugs the answer.
func TestSearchQueriesFrom_IgnoresOpenPage(t *testing.T) {
	search := responsesItem{
		Type: "web_search_call",
		Action: struct {
			Type   string   `json:"type"`
			Query  string   `json:"query"`
			Querys []string `json:"queries"`
		}{Type: "search", Query: "population of Singapore"},
	}
	openPage := responsesItem{
		Type: "web_search_call",
		Action: struct {
			Type   string   `json:"type"`
			Query  string   `json:"query"`
			Querys []string `json:"queries"`
		}{Type: "open_page", Query: "https://en.wikipedia.org/wiki/Demographics_of_Singapore"},
	}

	if got := searchQueriesFrom(search); len(got) != 1 || got[0] != "population of Singapore" {
		t.Errorf("search action = %v", got)
	}
	if got := searchQueriesFrom(openPage); len(got) != 0 {
		t.Errorf("open_page action = %v, want none; a page URL is not a search query", got)
	}
}

// TestResponsesToDomain_InterleavedReasoningAndSearches walks the exact item
// sequence the live Meta API returned: reasoning, message, search, reasoning,
// message, open_page, message.
func TestResponsesToDomain_InterleavedReasoningAndSearches(t *testing.T) {
	c, _ := newTestClient(t, "")
	msg := func(text string, anns ...responsesAnnotation) responsesItem {
		return responsesItem{Type: "message", Content: []responsesContent{
			{Type: "output_text", Text: text, Annotations: anns},
		}}
	}
	action := func(kind, q string) responsesItem {
		return responsesItem{Type: "web_search_call", Action: struct {
			Type   string   `json:"type"`
			Query  string   `json:"query"`
			Querys []string `json:"queries"`
		}{Type: kind, Query: q}}
	}

	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{
			Status: "completed",
			Output: []responsesItem{
				{Type: "reasoning"},
				msg("I'll search for that."),
				action("search", "population of Singapore 2026"),
				{Type: "reasoning"},
				msg("The results conflict, so I'll read the demographics page."),
				action("open_page", "https://en.wikipedia.org/wiki/Demographics_of_Singapore"),
				msg("6208500", responsesAnnotation{
					Type:  "url_citation",
					URL:   "https://en.wikipedia.org/wiki/Demographics_of_Singapore",
					Title: "Demographics of Singapore",
				}),
			},
		},
	)

	if resp.FinishReason != domain.FinishReasonComplete {
		t.Errorf("finish reason = %v (%s)", resp.FinishReason, resp.FinishDetail)
	}
	if !contains(resp.Completion, "6208500") {
		t.Errorf("completion = %q, want the final answer present", resp.Completion)
	}
	if len(resp.Citations) != 1 {
		t.Fatalf("citations = %+v, want 1", resp.Citations)
	}
	// Only the search action is a query; the open_page is navigation.
	if len(resp.SearchQueries) != 1 || resp.SearchQueries[0] != "population of Singapore 2026" {
		t.Errorf("search queries = %v", resp.SearchQueries)
	}
}

// TestResponsesToDomain_MessagesAreSeparateNotConcatenated guards the bug the
// live api.meta.ai run exposed: a reasoning model emits several `message`
// items — narration before each search, then the answer — and joining them
// produced a wall of text ending in a half-finished sentence with no answer and
// no citations. The answer is the LAST message; the rest is transcript.
func TestResponsesToDomain_MessagesAreSeparateNotConcatenated(t *testing.T) {
	c, _ := newTestClient(t, "")
	msg := func(text string, anns ...responsesAnnotation) responsesItem {
		return responsesItem{Type: "message", Content: []responsesContent{
			{Type: "output_text", Text: text, Annotations: anns},
		}}
	}

	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{
			Status: "completed",
			Output: []responsesItem{
				{Type: "reasoning"},
				msg("I'll look up Singapore's current population."),
				{Type: "web_search_call", Action: struct {
					Type   string   `json:"type"`
					Query  string   `json:"query"`
					Querys []string `json:"queries"`
				}{Type: "search", Query: "Singapore population 2026"}},
				msg("The results conflict, so I'll open the official release."),
				{Type: "web_search_call", Action: struct {
					Type   string   `json:"type"`
					Query  string   `json:"query"`
					Querys []string `json:"queries"`
				}{Type: "open_page", Query: "https://population.gov.sg"}},
				msg("6.21 million.", responsesAnnotation{
					Type: "url_citation", URL: "https://population.gov.sg/pib", Title: "Population in Brief",
				}),
			},
		},
	)

	// The answer alone, not the narration.
	if resp.Completion != "6.21 million." {
		t.Errorf("completion = %q, want only the final message", resp.Completion)
	}
	if len(resp.Messages) != 3 {
		t.Fatalf("messages = %d, want 3 preserved separately", len(resp.Messages))
	}
	// Citations belong to the message that made the claim, not the transcript.
	if len(resp.Messages[0].Citations) != 0 || len(resp.Messages[1].Citations) != 0 {
		t.Error("narration messages were given citations they do not support")
	}
	if len(resp.Messages[2].Citations) != 1 {
		t.Errorf("the answer's citations = %+v, want 1", resp.Messages[2].Citations)
	}
	// The flattened list still carries every citation.
	if len(resp.Citations) != 1 {
		t.Errorf("flattened citations = %+v, want 1", resp.Citations)
	}
}

func TestResponsesToDomain_SingleMessageHasNoMessagesList(t *testing.T) {
	// A non-reasoning model sends one message. The list is then redundant, and
	// the facade synthesises the single item from Completion instead.
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{
			Status: "completed",
			Output: []responsesItem{
				{Type: "message", Content: []responsesContent{{Type: "output_text", Text: "pong"}}},
			},
		},
	)
	if resp.Completion != "pong" {
		t.Errorf("completion = %q", resp.Completion)
	}
	if len(resp.Messages) != 1 {
		t.Errorf("messages = %d, want 1", len(resp.Messages))
	}
}

func TestResponsesToDomain_EmptyMessageItemsAreDropped(t *testing.T) {
	c, _ := newTestClient(t, "")
	resp := c.responsesToDomain(
		domain.VendorRequest{Target: groundedTarget("https://x.test/v1", &domain.Grounding{})},
		responsesReply{
			Status: "completed",
			Output: []responsesItem{
				{Type: "message", Content: []responsesContent{{Type: "refusal"}}},
				{Type: "message", Content: []responsesContent{{Type: "output_text", Text: "real answer"}}},
			},
		},
	)
	if len(resp.Messages) != 1 {
		t.Fatalf("messages = %d, want the empty one dropped", len(resp.Messages))
	}
	if resp.Completion != "real answer" {
		t.Errorf("completion = %q", resp.Completion)
	}
}
