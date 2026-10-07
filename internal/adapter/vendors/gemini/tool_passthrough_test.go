package gemini

import (
	"encoding/json"
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// ADR-177 — when ContentsJSON is present the body is built from it (multi-turn,
// incl. functionCall/functionResponse parts), NOT from the single Prompt.
func TestBuildGeminiBody_ContentsJSON(t *testing.T) {
	contents := `[
		{"role":"user","parts":[{"text":"what's the weather in SG?"}]},
		{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"city":"SG"}}}]},
		{"role":"user","parts":[{"functionResponse":{"name":"get_weather","response":{"temp":31}}}]}
	]`
	body := buildGeminiBody(domain.VendorRequest{
		Prompt:       "IGNORED when contents present",
		ContentsJSON: contents,
	})
	if len(body.Contents) != 3 {
		t.Fatalf("got %d contents, want 3 (built from ContentsJSON, not Prompt)", len(body.Contents))
	}
	if body.Contents[0].Parts[0].Text != "what's the weather in SG?" {
		t.Errorf("content[0] text = %q", body.Contents[0].Parts[0].Text)
	}
	if body.Contents[1].Parts[0].FunctionCall == nil || body.Contents[1].Parts[0].FunctionCall.Name != "get_weather" {
		t.Errorf("content[1] functionCall not threaded: %+v", body.Contents[1].Parts[0])
	}
	if body.Contents[2].Parts[0].FunctionResponse == nil || body.Contents[2].Parts[0].FunctionResponse.Name != "get_weather" {
		t.Errorf("content[2] functionResponse not threaded: %+v", body.Contents[2].Parts[0])
	}
}

// ADR-177 — ToolsJSON is passed through as the gemini `tools` field.
func TestBuildGeminiBody_ToolsJSON(t *testing.T) {
	tools := `[{"functionDeclarations":[{"name":"get_weather","description":"weather","parameters":{"type":"object"}}]}]`
	body := buildGeminiBody(domain.VendorRequest{
		Prompt:    "hi",
		ToolsJSON: tools,
	})
	if len(body.Tools) == 0 {
		t.Fatalf("body.Tools empty — ToolsJSON not threaded")
	}
	// Round-trips to the gemini tools wire shape.
	raw, _ := json.Marshal(body.Tools)
	var probe []map[string]any
	if err := json.Unmarshal(raw, &probe); err != nil {
		t.Fatalf("tools not valid json: %v", err)
	}
	if _, ok := probe[0]["functionDeclarations"]; !ok {
		t.Errorf("tools missing functionDeclarations: %s", raw)
	}
}

// ADR-177 — legacy text-only path unchanged: no contents/tools ⇒ single user
// prompt, no tools field.
func TestBuildGeminiBody_LegacyPromptPath(t *testing.T) {
	body := buildGeminiBody(domain.VendorRequest{Prompt: "hello"})
	if len(body.Contents) != 1 || body.Contents[0].Parts[0].Text != "hello" {
		t.Fatalf("legacy prompt path broken: %+v", body.Contents)
	}
	if len(body.Tools) != 0 {
		t.Errorf("legacy path set tools: %+v", body.Tools)
	}
}

// ADR-177 — a response carrying functionCall parts surfaces ToolCallsJSON +
// a non-terminal finish (no completion text required).
func TestGeminiResponse_ToDomain_ToolCalls(t *testing.T) {
	raw := `{
		"candidates":[{"content":{"role":"model","parts":[
			{"functionCall":{"name":"get_weather","args":{"city":"SG"}}}
		]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":3},
		"modelVersion":"gemini-2.5-flash"
	}`
	var gr geminiResponse
	if err := json.Unmarshal([]byte(raw), &gr); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	resp, err := gr.toDomain("gemini-2.5-flash")
	if err != nil {
		t.Fatalf("toDomain: %v", err)
	}
	if resp.ToolCallsJSON == "" {
		t.Fatalf("ToolCallsJSON empty — functionCall not surfaced")
	}
	var calls []map[string]any
	if err := json.Unmarshal([]byte(resp.ToolCallsJSON), &calls); err != nil {
		t.Fatalf("ToolCallsJSON not valid json: %v", err)
	}
	if len(calls) != 1 || calls[0]["name"] != "get_weather" {
		t.Errorf("tool calls = %s, want [get_weather]", resp.ToolCallsJSON)
	}
}

// ADR-177 — a plain text response has empty ToolCallsJSON (regression guard).
func TestGeminiResponse_ToDomain_TextHasNoToolCalls(t *testing.T) {
	raw := `{"candidates":[{"content":{"role":"model","parts":[{"text":"31C and sunny"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4}}`
	var gr geminiResponse
	_ = json.Unmarshal([]byte(raw), &gr)
	resp, err := gr.toDomain("gemini-2.5-flash")
	if err != nil {
		t.Fatalf("toDomain: %v", err)
	}
	if resp.ToolCallsJSON != "" {
		t.Errorf("ToolCallsJSON = %q, want empty for text response", resp.ToolCallsJSON)
	}
	if resp.Completion != "31C and sunny" {
		t.Errorf("completion = %q", resp.Completion)
	}
}
