package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// ---------------------------------------------------------------------------
// POST /v1/chat/completions and /v1/completions
// ---------------------------------------------------------------------------

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	body, err := decodeBody(r, maxChatBodyBytes)
	if err != nil {
		if strings.Contains(err.Error(), "too large") {
			writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "body_too_large", err.Error())
			return
		}
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}

	if rawBool(body, "stream") {
		writeError(w, http.StatusNotImplemented, "invalid_request_error", "streaming_unsupported",
			"this gateway does not implement SSE streaming; retry with \"stream\": false")
		return
	}

	modelName := rawString(body, "model")
	if modelName == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter",
			"\"model\" is required and must name an entry in the model registry")
		return
	}
	if _, ok := s.registry.Get(modelName); !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model \"%s\" is not in the registry; GET /v1/models lists what is", modelName))
		return
	}

	// Split system/developer messages out of the conversation; the remainder
	// is the conversation that rides contents_json.
	messages, ok := rawArray(body, "messages")
	if !ok || len(messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter",
			"\"messages\" must contain at least one message")
		return
	}

	systemParts, conversation := splitSystemMessages(messages)
	if len(conversation) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "empty_conversation",
			"messages contains only a system prompt; add at least one user message")
		return
	}

	// Grounding: when the tools declare web_search/web_fetch, the model must
	// advertise web_search + have a grounding block, and the surface must be
	// chat_completions (else the caller must use /v1/responses).
	tools, _ := rawArray(body, "tools")
	grounded := false
	if wantsGrounding(tools) {
		surface, gerr, ok := s.groundingSurfaceFor(modelName)
		if !ok {
			writeError(w, gerr.status, gerr.errType, gerr.code, gerr.message)
			return
		}
		if surface != "chat_completions" {
			writeError(w, http.StatusBadRequest, "invalid_request_error", "wrong_grounding_surface",
				fmt.Sprintf("model \"%s\" is configured to ground on the %s surface; use POST /v1/responses for hosted web search",
					modelName, surface))
			return
		}
		grounded = true
	}

	id := s.extractIdentity(r)
	req := s.buildInvokeRequest(r, id, modelName, body, "openai_compat")
	req.ResponseModality = "TEXT"
	if grounded {
		req.ResponseModality = "GROUNDED"
	}
	req.SystemPrompt = strings.Join(systemParts, "\n\n")
	req.ContentsJSON = mustMarshal(conversation)
	if len(tools) > 0 {
		req.ToolsJSON = mustMarshal(tools)
	}

	resp, err := s.invoker.Execute(r.Context(), req)
	if err != nil {
		m := mapInvokeError(err)
		writeError(w, m.status, m.errType, m.code, m.message)
		return
	}
	if m, ok := mapInbandFinish(resp); ok {
		writeError(w, m.status, m.errType, m.code, m.message)
		return
	}

	writeJSON(w, http.StatusOK, s.chatCompletionResponse(resp, grounded))
}

// splitSystemMessages separates system/developer messages from the
// conversation, returning the system text parts and the remaining messages.
// System message content is extracted to plain text (string or text parts).
func splitSystemMessages(messages []json.RawMessage) ([]string, []json.RawMessage) {
	var systemParts []string
	var conversation []json.RawMessage
	for _, raw := range messages {
		var msg struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		if err := json.Unmarshal(raw, &msg); err != nil {
			conversation = append(conversation, raw)
			continue
		}
		role := strings.ToLower(msg.Role)
		if role == "system" || role == "developer" {
			if text := rawToText(msg.Content); text != "" {
				systemParts = append(systemParts, text)
			}
		} else {
			conversation = append(conversation, raw)
		}
	}
	return systemParts, conversation
}

// rawToText extracts plain text from a message content field (string or
// array of text parts), mirroring the Python facade's _raw_to_text.
func rawToText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			sb.WriteString(p.Text)
		}
		return sb.String()
	}
	return ""
}

// ---------------------------------------------------------------------------
// POST /v1/responses
// ---------------------------------------------------------------------------

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	body, err := decodeBody(r, maxChatBodyBytes)
	if err != nil {
		if strings.Contains(err.Error(), "too large") {
			writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "body_too_large", err.Error())
			return
		}
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}

	if rawBool(body, "stream") {
		writeError(w, http.StatusNotImplemented, "invalid_request_error", "streaming_unsupported",
			"this gateway does not implement SSE streaming; retry with \"stream\": false")
		return
	}

	modelName := rawString(body, "model")
	if modelName == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter",
			"\"model\" is required and must name an entry in the model registry")
		return
	}
	if _, ok := s.registry.Get(modelName); !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model \"%s\" is not in the registry; GET /v1/models lists what is", modelName))
		return
	}

	// Grounding check — the responses surface is the hosted-web-search
	// surface, so a web_search tool requires the model to be grounded.
	tools, _ := rawArray(body, "tools")
	grounded := wantsGrounding(tools)
	if grounded {
		_, gerr, ok := s.groundingSurfaceFor(modelName)
		if !ok {
			writeError(w, gerr.status, gerr.errType, gerr.code, gerr.message)
			return
		}
	}

	// Flatten the input: a string becomes the prompt; an array becomes the
	// conversation (contents_json).
	prompt, contents, ierr := flattenResponsesInput(body["input"])
	if ierr != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_input", ierr.Error())
		return
	}

	id := s.extractIdentity(r)
	req := s.buildInvokeRequest(r, id, modelName, body, "responses")
	req.ResponseModality = "TEXT"
	if grounded {
		req.ResponseModality = "GROUNDED"
	}
	req.Prompt = prompt
	req.SystemPrompt = rawString(body, "instructions")
	if contents != "" {
		req.ContentsJSON = contents
	}
	if len(tools) > 0 {
		req.ToolsJSON = mustMarshal(tools)
	}

	resp, err := s.invoker.Execute(r.Context(), req)
	if err != nil {
		m := mapInvokeError(err)
		writeError(w, m.status, m.errType, m.code, m.message)
		return
	}
	if m, ok := mapInbandFinish(resp); ok {
		writeError(w, m.status, m.errType, m.code, m.message)
		return
	}

	writeJSON(w, http.StatusOK, s.responsesResponse(resp, grounded))
}

// flattenResponsesInput converts the OpenAI Responses `input` field into a
// prompt string + contents JSON, mirroring the Python compat.
// flatten_responses_input. A string input becomes the prompt; a single-item
// array becomes the prompt; a multi-item array becomes the conversation.
func flattenResponsesInput(raw json.RawMessage) (prompt, contents string, err error) {
	if len(raw) == 0 {
		return "", "", fmt.Errorf("\"input\" is required")
	}

	// String input → prompt.
	var s string
	if json.Unmarshal(raw, &s) == nil {
		if strings.TrimSpace(s) == "" {
			return "", "", fmt.Errorf("\"input\" must not be empty")
		}
		return s, "", nil
	}

	// Array input.
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return "", "", fmt.Errorf("\"input\" must be a string or an array of input items")
	}
	if len(items) == 0 {
		return "", "", fmt.Errorf("\"input\" must not be empty")
	}
	if len(items) == 1 {
		text, err := inputItemText(items[0])
		if err != nil {
			return "", "", err
		}
		return text, "", nil
	}

	// Multi-item array → conversation messages.
	conversation := make([]map[string]any, 0, len(items))
	for i, item := range items {
		var msg map[string]any
		if err := json.Unmarshal(item, &msg); err != nil {
			return "", "", fmt.Errorf("input[%d] must be an object", i)
		}
		text, err := inputItemText(item)
		if err != nil {
			return "", "", err
		}
		role, _ := msg["role"].(string)
		if role == "" {
			role = "user"
		}
		conversation = append(conversation, map[string]any{"role": role, "content": text})
	}
	return "", mustMarshal(conversation), nil
}

// inputItemText extracts the text content from one Responses input item,
// mirroring the Python compat._input_item_text.
func inputItemText(item json.RawMessage) (string, error) {
	var msg map[string]any
	if err := json.Unmarshal(item, &msg); err != nil {
		return "", fmt.Errorf("each entry of \"input\" must be an object")
	}

	content, ok := msg["content"]
	if !ok {
		if text, ok := msg["text"].(string); ok && text != "" {
			return text, nil
		}
		return "", fmt.Errorf("an input item needs content or text")
	}

	switch c := content.(type) {
	case string:
		if c == "" {
			return "", fmt.Errorf("the input item carried no text")
		}
		return c, nil
	case []any:
		var sb strings.Builder
		for _, part := range c {
			p, ok := part.(map[string]any)
			if !ok {
				continue
			}
			kind, _ := p["type"].(string)
			if kind != "" && kind != "input_text" && kind != "text" && kind != "output_text" {
				return "", fmt.Errorf("unsupported non-text input part")
			}
			if text, ok := p["text"].(string); ok {
				sb.WriteString(text)
			}
		}
		if sb.Len() == 0 {
			return "", fmt.Errorf("the input item carried no text")
		}
		return sb.String(), nil
	default:
		return "", fmt.Errorf("invalid input item content")
	}
}

// ---------------------------------------------------------------------------
// POST /v1/images/generations and /v1/images/edits
// ---------------------------------------------------------------------------

func (s *Server) handleImages(w http.ResponseWriter, r *http.Request) {
	body, err := decodeBody(r, maxImageBodyBytes)
	if err != nil {
		if strings.Contains(err.Error(), "too large") {
			writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "body_too_large", err.Error())
			return
		}
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}

	prompt := rawString(body, "prompt")
	if prompt == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter", "\"prompt\" is required")
		return
	}

	modelName := rawString(body, "model")
	if modelName == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter",
			"\"model\" is required and must name an image-capable entry in the model registry")
		return
	}
	if _, ok := s.registry.Get(modelName); !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model \"%s\" is not in the registry; GET /v1/models lists what is", modelName))
		return
	}

	if format := strings.ToLower(rawString(body, "response_format")); format == "url" {
		writeError(w, http.StatusNotImplemented, "invalid_request_error", "url_response_unsupported",
			"this gateway returns bytes, not hosted URLs; use \"response_format\": \"b64_json\" (the default)")
		return
	}

	id := s.extractIdentity(r)
	req := s.buildInvokeRequest(r, id, modelName, body, "openai_compat")
	req.ResponseModality = "IMAGE"
	req.Prompt = prompt
	// Image-specific params: size / quality / style.
	imgParams := map[string]any{}
	for _, key := range []string{"size", "quality", "style"} {
		if raw, ok := body[key]; ok {
			var v any
			if err := json.Unmarshal(raw, &v); err == nil && v != nil {
				imgParams[key] = v
			}
		}
	}
	if len(imgParams) > 0 {
		req.GenerationConfig = imgParams
	}

	resp, err := s.invoker.Execute(r.Context(), req)
	if err != nil {
		m := mapInvokeError(err)
		writeError(w, m.status, m.errType, m.code, m.message)
		return
	}
	if m, ok := mapInbandFinish(resp); ok {
		writeError(w, m.status, m.errType, m.code, m.message)
		return
	}

	if resp.FinishReason != domain.FinishReasonComplete || len(resp.ImageBytes) == 0 {
		// Sanitized: the raw FinishDetail may include the upstream error body.
		writeError(w, http.StatusBadGateway, "upstream_error", "no_image",
			"the model returned no image")
		return
	}

	writeJSON(w, http.StatusOK, s.imageResponse(resp))
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings
// ---------------------------------------------------------------------------

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	body, err := decodeBody(r, maxEmbedBodyBytes)
	if err != nil {
		if strings.Contains(err.Error(), "too large") {
			writeError(w, http.StatusRequestEntityTooLarge, "invalid_request_error", "body_too_large", err.Error())
			return
		}
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}

	modelName := rawString(body, "model")
	if modelName == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter", "\"model\" is required")
		return
	}
	if _, ok := s.registry.Get(modelName); !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model \"%s\" is not in the registry", modelName))
		return
	}

	if format := strings.ToLower(rawString(body, "encoding_format")); format == "base64" {
		writeError(w, http.StatusNotImplemented, "invalid_request_error", "base64_unsupported",
			"\"encoding_format\": \"base64\" is not implemented; omit it or send \"float\"")
		return
	}

	inputs, ierr := embeddingInputs(body["input"])
	if ierr != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_parameter", ierr.Error())
		return
	}

	id := s.extractIdentity(r)
	dimensions := int32(0)
	if raw, ok := body["dimensions"]; ok {
		var d int64
		if err := json.Unmarshal(raw, &d); err == nil && d > 0 {
			dimensions = int32(d)
		}
	}

	rows := make([]map[string]any, 0, len(inputs))
	var total int64
	for i, inputText := range inputs {
		resp, err := s.embedder.Embed(r.Context(), domain.EmbedFlowRequest{
			TenantID:         id.tenantID,
			GCID:             id.gcid,
			AgentID:          id.agentID,
			LogicalModelID:   domain.LogicalModelID(modelName),
			Text:             inputText,
			OutputDimensions: dimensions,
			Traceparent:      id.traceparent,
			Tracestate:       id.tracestate,
		})
		if err != nil {
			m := mapEmbedError(err)
			writeError(w, m.status, m.errType, m.code, m.message)
			return
		}
		rows = append(rows, map[string]any{
			"object":    "embedding",
			"index":     i,
			"embedding": resp.Values,
		})
		total += resp.Usage.InputTokens
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"model":  modelName,
		"data":   rows,
		"usage": map[string]any{
			"prompt_tokens":     total,
			"completion_tokens": 0,
			"total_tokens":      total,
		},
	})
}

// embeddingInputs converts the OpenAI embeddings `input` field into a list
// of strings, mirroring the Python compat.embedding_inputs.
func embeddingInputs(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("\"input\" is required")
	}

	// String input.
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return nil, fmt.Errorf("\"input\" must not be empty")
		}
		return []string{s}, nil
	}

	// Array of strings.
	var items []string
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("\"input\" must be a string or an array of strings; token arrays are not supported")
	}
	if len(items) == 0 {
		return nil, fmt.Errorf("\"input\" must not be empty")
	}
	for i, item := range items {
		if item == "" {
			return nil, fmt.Errorf("\"input\"[%d] is empty", i)
		}
	}
	return items, nil
}

// mapEmbedError translates a domain embed failure into the OpenAI error
// envelope, mirroring the Python gateway's embed arm.
// mapEmbedError translates a domain embed failure into the OpenAI error
// envelope, mirroring the Python gateway's embed arm.
//
// ERROR SANITIZATION: the client-facing message NEVER includes the raw
// upstream error body. The upstream body may contain internal hostnames,
// stack traces, or other sensitive information. Gateway-controlled error
// messages (validation refusals, ledger failures) are safe to relay.
func mapEmbedError(err error) mappedError {
	if us := domain.UpstreamStatusOf(err); us != nil {
		return mappedError{
			status:  *us,
			errType: upstreamErrorType(*us),
			code:    "upstream_error",
			message: domain.SanitizeUpstreamError(*us, ""),
		}
	}
	// For non-upstream errors, the gateway-controlled message is safe to
	// relay. The full error (with the upstream body) is available in the
	// logs for internal debugging.
	msg := err.Error()
	switch {
	case strings.Contains(msg, "required") || strings.Contains(msg, "not an embedding model"):
		return mappedError{http.StatusBadRequest, "invalid_request_error", "invalid_request", msg}
	case strings.Contains(msg, "vendor dispatch"):
		return mappedError{http.StatusBadGateway, "upstream_error", "vendor_error", msg}
	case strings.Contains(msg, "ledger enqueue failed"):
		return mappedError{http.StatusInternalServerError, "upstream_error", "internal_error", msg}
	default:
		return mappedError{http.StatusInternalServerError, "upstream_error", "internal_error", msg}
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// mustMarshal JSON-encodes v, returning "" on error. Used for request fields
// that were already validated as JSON.
func mustMarshal(v any) string {
	data, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(data)
}
