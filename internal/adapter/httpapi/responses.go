// responses.go — the OpenAI **Responses** API surface.
//
// This exists because hosted web search is not reachable through
// /v1/chat/completions on OpenAI: the tool lives on /v1/responses. The
// aggregators (Meta's api.meta.ai among them) follow the same convention, so a
// caller holding one of those credentials needs this endpoint to exist here or
// the request has nowhere to go.
//
// The gateway normalises rather than proxies. A caller says "web_search" in the
// abstract; the registry decides whether that becomes OpenAI's
// {"type":"web_search"} on /responses, Anthropic's web_search_20250305 on
// /messages, or something a self-hosted server invented. So a client written
// against the gateway keeps working when you swap providers.
//
// Budget, ledger and the fallback chain all still apply — a grounded call is a
// normal dispatch that happens to use a different endpoint.
package httpapi

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// responsesRequest is the request body. Only the fields the gateway acts on are
// modelled; anything else is ignored rather than rejected, because the
// Responses API surface grows and a client sending a field this build has
// never heard of should still get an answer.
type responsesRequest struct {
	Model  string `json:"model"`
	Input  any    `json:"input"`
	Stream bool   `json:"stream"`

	// Tools is where the search request arrives. Any entry whose type contains
	// "web_search" or "web_fetch" counts as a grounding request.
	Tools []struct {
		Type string          `json:"type"`
		Name string          `json:"name"`
		Max  int             `json:"max_uses"`
		Raw  json.RawMessage `json:"-"`
	} `json:"tools"`

	// Instructions is the Responses API's system-prompt field.
	Instructions string `json:"instructions"`

	// Generation knobs are captured loosely so provider extensions survive.
	Temperature     *float64 `json:"temperature"`
	TopP            *float64 `json:"top_p"`
	MaxOutput       *int     `json:"max_output_tokens"`
	MaxTokens       *int     `json:"max_tokens"`
	Seed            *int     `json:"seed"`
	ReasoningEffort string   `json:"reasoning_effort"`

	User string `json:"user"`
}

// wantsGrounding reports whether the caller asked for hosted web search.
//
// Matching is a substring test on the tool type because the providers disagree
// on the exact spelling — "web_search", "web_search_preview",
// "web_search_2025_08_26", "web_search_20250305" — and a client that names any
// of them means the same thing. A caller that names none is a plain call.
func (r responsesRequest) wantsGrounding() bool {
	for _, tool := range r.Tools {
		t := strings.ToLower(tool.Type)
		if strings.Contains(t, "web_search") || strings.Contains(t, "web_fetch") {
			return true
		}
	}
	return false
}

// responsesResponse is the reply. It keeps the provider's own item array — a
// Responses client parses it by type — and adds a normalised citation list
// under the gateway's own key so a caller does not need a branch per vendor.
type responsesResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created_at"`
	Model   string `json:"model"`
	Status  string `json:"status"`

	// Output mirrors the upstream shape: a heterogeneous array of items the
	// client parses BY TYPE. Passed through with the provider's own types, so
	// a reasoning model's `reasoning` items and a `web_search_call` survive.
	Output []map[string]any `json:"output"`

	Usage *chatUsage `json:"usage,omitempty"`

	// ChoraGateway is additive and ignorable by any Responses client.
	ChoraGateway *groundedMeta `json:"chora_gateway,omitempty"`
}

type groundedMeta struct {
	Vendor           string                     `json:"vendor"`
	Grounded         bool                       `json:"grounded"`
	Citations        []domain.GroundingCitation `json:"citations"`
	SearchQueries    []string                   `json:"search_queries,omitempty"`
	FallbackChain    []string                   `json:"fallback_chain,omitempty"`
	LatencyMs        int32                      `json:"latency_ms"`
	InvocationID     string                     `json:"invocation_id"`
	GroundingSurface string                     `json:"grounding_surface,omitempty"`

	// FinishReason / FinishDetail mirror the gateway's own verdict, which is
	// NOT always the provider's `status`. A truncated research run comes back
	// `status: incomplete` carrying narration and no answer; reporting that as
	// "completed" tells the caller the model finished when it was cut off
	// mid-search.
	FinishReason string `json:"finish_reason,omitempty"`
	FinishDetail string `json:"finish_detail,omitempty"`

	// Messages is the ordered assistant messages with their per-message
	// citations. A reasoning model that searched narrates first and answers
	// last, and this is what tells a caller which text is the answer.
	Messages []domain.GroundingMessage `json:"messages,omitempty"`
}

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", "POST only")
		return
	}

	var req responsesRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}
	if req.Stream {
		writeError(w, http.StatusNotImplemented, "invalid_request_error", "streaming_unsupported",
			"this gateway does not implement SSE streaming; retry with \"stream\": false")
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter",
			"\"model\" is required and must name an entry in the model registry")
		return
	}
	if !s.models.Known(req.Model) {
		writeError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model %q is not in the registry; GET /v1/models lists what is", req.Model))
		return
	}

	// Grounding is checked HERE rather than left to the domain, so an
	// ungroundable request costs no round trip and returns an actionable 400
	// naming the fix instead of a 500 from deep in the dispatch chain.
	if req.wantsGrounding() {
		if _, gErr := s.groundingSurfaceFor(req.Model); gErr != nil {
			slog.WarnContext(r.Context(), "httpapi: grounding refused",
				"model", req.Model, "reason", gErr)
			writeError(w, http.StatusBadRequest, "invalid_request_error", "grounding_unavailable", gErr.Error())
			return
		}
	}

	invoke, err := s.responsesToInvoke(r, req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_input", err.Error())
		return
	}

	resp, err := s.gw.Invoke(r.Context(), invoke)
	if err != nil {
		s.writeGatewayError(w, r, err)
		return
	}
	if refuseIfBudgetBlocked(w, r, req.Model, resp) {
		return
	}

	out := responsesResponse{
		ID:      resp.InvocationID,
		Object:  "response",
		Created: timeToUnix(resp.CompletedAt),
		Model:   resp.ModelVersion,
		Status:  "completed",
		Output:  buildResponsesOutput(resp, invoke.ResponseModality),
		Usage: &chatUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
		ChoraGateway: &groundedMeta{
			Vendor:           resp.Vendor,
			Grounded:         invoke.ResponseModality == domain.ModalityGrounded,
			Citations:        resp.Citations,
			SearchQueries:    resp.SearchQueries,
			FallbackChain:    resp.FallbackChain,
			LatencyMs:        resp.LatencyMs,
			InvocationID:     resp.InvocationID,
			GroundingSurface: resp.GroundingSurface,
			FinishReason:     resp.FinishReason.String(),
			FinishDetail:     resp.FinishDetail,
			Messages:         resp.Messages,
		},
	}
	writeJSON(w, http.StatusOK, out)
}

// responsesToInvoke converts the request body into the domain shape.
func (s *Server) responsesToInvoke(r *http.Request, req responsesRequest) (domain.InvokeRequest, error) {
	invoke := s.baseRequest(r, req.Model)
	invoke.ActionCode = firstNonEmpty(req.User, s.defaultAgentID)
	invoke.SystemPrompt = req.Instructions

	prompt, contents, err := flattenResponsesInput(req.Input)
	if err != nil {
		return domain.InvokeRequest{}, err
	}
	invoke.Prompt = prompt
	invoke.ContentsJSON = contents

	if req.wantsGrounding() {
		invoke.ResponseModality = domain.ModalityGrounded
	}

	invoke.GenerationConfig = map[string]any{}
	if req.Temperature != nil {
		invoke.GenerationConfig["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		invoke.GenerationConfig["top_p"] = *req.TopP
	}
	if req.MaxOutput != nil {
		invoke.GenerationConfig["max_output_tokens"] = float64(*req.MaxOutput)
	}
	if req.MaxTokens != nil {
		invoke.GenerationConfig["max_tokens"] = float64(*req.MaxTokens)
	}
	if req.Seed != nil {
		invoke.GenerationConfig["seed"] = float64(*req.Seed)
	}
	if req.ReasoningEffort != "" {
		invoke.GenerationConfig["reasoning_effort"] = req.ReasoningEffort
	}
	return invoke, nil
}

// flattenResponsesInput normalises the `input` union into either a bare prompt
// or a full conversation.
//
// `input` is a string OR an array of input items, and the item shape is close
// to but not identical to a chat message. A single-item array collapses to the
// string form; anything longer is forwarded as a conversation, which is what the
// vendor adapter expects.
func flattenResponsesInput(input any) (prompt, contents string, err error) {
	switch v := input.(type) {
	case nil:
		return "", "", errors.New("\"input\" is required")
	case string:
		if strings.TrimSpace(v) == "" {
			return "", "", errors.New("\"input\" must not be empty")
		}
		return v, "", nil
	case []any:
		if len(v) == 0 {
			return "", "", errors.New("\"input\" must not be empty")
		}
		if len(v) == 1 {
			// A one-item array is just the scalar form spelled out.
			return inputItemToText(v[0])
		}
		msgs, err := responsesItemsToMessages(v)
		if err != nil {
			return "", "", err
		}
		encoded, mErr := json.Marshal(msgs)
		if mErr != nil {
			return "", "", fmt.Errorf("could not normalise input items: %w", mErr)
		}
		return "", string(encoded), nil
	default:
		return "", "", errors.New("\"input\" must be a string or an array of input items")
	}
}

// inputItemToText flattens a single input item into plain text.
func inputItemToText(item any) (string, string, error) {
	m, ok := item.(map[string]any)
	if !ok {
		return "", "", errors.New("each entry of \"input\" must be an object")
	}
	content, present := m["content"]
	if !present {
		// A bare string item.
		if s, ok := m["text"].(string); ok {
			return s, "", nil
		}
		return "", "", errors.New("an input item needs `content` or `text`")
	}
	switch c := content.(type) {
	case string:
		return c, "", nil
	case []any:
		var b strings.Builder
		for _, part := range c {
			pm, ok := part.(map[string]any)
			if !ok {
				continue
			}
			switch pm["type"] {
			case "input_text", "text", "output_text", "":
				if txt, ok := pm["text"].(string); ok {
					b.WriteString(txt)
				}
			default:
				// A non-text part (image, file) cannot be flattened into a
				// prompt string; say so rather than silently dropping content.
				return "", "", fmt.Errorf(
					"input part type %q is not supported on this surface; use /v1/chat/completions for multimodal input",
					pm["type"])
			}
		}
		if b.Len() == 0 {
			return "", "", errors.New("the input item carried no text")
		}
		return b.String(), "", nil
	default:
		return "", "", errors.New("an input item's `content` must be a string or an array of parts")
	}
}

// responsesItemsToMessages converts input items into the chat-message shape the
// vendor adapters speak.
func responsesItemsToMessages(items []any) ([]chatMessage, error) {
	out := make([]chatMessage, 0, len(items))
	for i, item := range items {
		m, ok := item.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("input[%d] must be an object", i)
		}
		role, _ := m["role"].(string)
		if role == "" {
			role = "user"
		}
		text, _, err := inputItemToText(m)
		if err != nil {
			return nil, fmt.Errorf("input[%d]: %w", i, err)
		}
		out = append(out, chatMessage{Role: role, Content: mustJSON(text)})
	}
	return out, nil
}

// buildResponsesOutput reconstructs the provider's output-item array from the
// normalised domain response.
//
// The gateway does not keep the raw upstream item array (the domain carries the
// text and the citations, not provider-specific items), so this rebuilds the
// shape a Responses client expects: a `web_search_call` item when a search ran,
// then a `message` item with `output_text` carrying the answer and its
// annotations.
//
// Consequence worth stating plainly: a reasoning model's `reasoning` items are
// NOT reproduced here. The gateway does not model them, so a client that
// wanted them gets the answer and citations but no visible chain of thought.
func buildResponsesOutput(resp domain.InvokeResponse, modality string) []map[string]any {
	items := make([]map[string]any, 0, 2)

	if modality == domain.ModalityGrounded {
		search := map[string]any{
			"type":   "web_search_call",
			"id":     "ws_" + resp.InvocationID,
			"status": "completed",
		}
		if len(resp.SearchQueries) > 0 {
			action := map[string]any{"type": "search"}
			if len(resp.SearchQueries) == 1 {
				action["query"] = resp.SearchQueries[0]
			} else {
				action["queries"] = resp.SearchQueries
			}
			search["action"] = action
		}
		items = append(items, search)
	}

	// One `message` item per upstream message, each carrying its OWN
	// annotations. Collapsing a reasoning model's narration and its answer into
	// a single output_text loses the distinction between "here is what I am
	// doing" and "here is the answer", and merges citations onto text they do
	// not support.
	messages := resp.Messages
	if len(messages) == 0 {
		// A single-message reply (or chat completions) has no Messages list;
		// synthesise one so the output shape is identical either way.
		messages = []domain.GroundingMessage{{
			Text:      resp.Completion,
			Citations: resp.Citations,
		}}
	}
	for i, msg := range messages {
		items = append(items, map[string]any{
			"type":   "message",
			"id":     fmt.Sprintf("msg_%s_%d", resp.InvocationID, i),
			"status": "completed",
			"role":   "assistant",
			"content": []map[string]any{{
				"type":        "output_text",
				"text":        msg.Text,
				"annotations": annotationsToJSON(msg.Citations),
			}},
		})
	}
	return items
}

// annotationsToJSON renders normalised citations as OpenAI url_citation
// annotations. An empty citation list still produces an empty array, never an
// omitted key: a caller needs to tell "grounded, nothing citable" from
// "not a grounded call at all".
func annotationsToJSON(citations []domain.GroundingCitation) []map[string]any {
	out := make([]map[string]any, 0, len(citations))
	for _, c := range citations {
		a := map[string]any{"type": "url_citation", "url": c.URL}
		if c.Title != "" {
			a["title"] = c.Title
		}
		// Offsets are omitted when the provider did not supply them (Anthropic
		// never does); a 0..0 range would read as "cited at the very start".
		if c.StartIndex > 0 || c.EndIndex > 0 {
			a["start_index"] = c.StartIndex
			a["end_index"] = c.EndIndex
		}
		out = append(out, a)
	}
	return out
}

// handleChatCompletionsGrounding is the /v1/chat/completions path for a caller
// that asked for search on the classic surface. It shares the chat handler and
// differs only in setting the modality, which the handler already understands.
func logGroundedCall(r *http.Request, model, surface string, citations int) {
	slog.InfoContext(r.Context(), "httpapi: grounded call",
		"model", model, "surface", surface, "citations", citations)
}

var errNoGrounding = errors.New("this gateway has no hosted web search configured for that model")
