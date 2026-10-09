package openai

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// ---------------------------------------------------------------------------
// Identity extraction
// ---------------------------------------------------------------------------

// identity carries the per-request identity + trace context extracted from
// the HTTP request. Mirrors the Python facade's attributes() function.
type identity struct {
	tenantID   string
	gcid       string
	agentID    string
	traceparent string
	tracestate  string
}

// extractIdentity pulls tenant/gcid from headers (x-chora-tenant-id,
// x-chora-gcid) with query-param fallback (tenant=), then falls back to the
// configured defaults. traceparent/tracestate come from headers only.
func (s *Server) extractIdentity(r *http.Request) identity {
	tenantID := r.Header.Get("x-chora-tenant-id")
	if tenantID == "" {
		tenantID = r.URL.Query().Get("tenant")
	}
	if tenantID == "" {
		tenantID = s.settings.DefaultTenantID
	}

	gcid := r.Header.Get("x-chora-gcid")
	if gcid == "" {
		gcid = s.settings.DefaultGCID
	}

	return identity{
		tenantID:    tenantID,
		gcid:        gcid,
		agentID:     s.settings.DefaultAgentID,
		traceparent: r.Header.Get("traceparent"),
		tracestate:  r.Header.Get("tracestate"),
	}
}

// ---------------------------------------------------------------------------
// Request body decoding
// ---------------------------------------------------------------------------

// Body size limits per route. These are hard caps enforced BEFORE JSON
// parsing — a request body larger than the limit is rejected with 413
// rather than being buffered into memory. The limits are generous enough
// for any legitimate request (a chat completion with a large conversation,
// an image generation with a long prompt) while bounding the memory an
// attacker can force the gateway to allocate.
const (
	// maxChatBodyBytes is the 10 MiB cap for chat-completions and
	// responses requests. A legitimate chat conversation with base64
	// inline images can approach this; anything larger is almost
	// certainly an abuse attempt.
	maxChatBodyBytes = 10 << 20
	// maxImageBodyBytes is the 50 MiB cap for image-generation requests.
	// Image prompts are short, but the limit is generous to accommodate
	// future multimodal inputs.
	maxImageBodyBytes = 50 << 20
	// maxEmbedBodyBytes is the 10 MiB cap for embedding requests.
	maxEmbedBodyBytes = 10 << 20
)

// decodeBody reads + parses the request body as a JSON object, enforcing a
// hard size limit. Returns an error suitable for the 400 invalid_json
// envelope when the body is not a JSON object, and a 413 error when the
// body exceeds the limit.
func decodeBody(r *http.Request, maxBytes int64) (map[string]json.RawMessage, error) {
	if r.Body == nil {
		return nil, fmt.Errorf("request body is required")
	}
	// http.MaxBytesReader enforces the limit DURING reading — a body larger
	// than maxBytes causes the decoder to fail with a clear error rather
	// than buffering the whole body into memory.
	r.Body = http.MaxBytesReader(nil, r.Body, maxBytes)
	var body map[string]json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		if strings.Contains(err.Error(), "request body too large") {
			return nil, fmt.Errorf("request body too large (max %d bytes)", maxBytes)
		}
		return nil, fmt.Errorf("could not parse request body: %v", err)
	}
	return body, nil
}

// rawString extracts a string field from the raw body map.
func rawString(body map[string]json.RawMessage, key string) string {
	raw, ok := body[key]
	if !ok {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return ""
	}
	return s
}

// rawBool extracts a boolean field from the raw body map.
func rawBool(body map[string]json.RawMessage, key string) bool {
	raw, ok := body[key]
	if !ok {
		return false
	}
	var b bool
	if err := json.Unmarshal(raw, &b); err != nil {
		return false
	}
	return b
}

// rawArray extracts an array field from the raw body map.
func rawArray(body map[string]json.RawMessage, key string) ([]json.RawMessage, bool) {
	raw, ok := body[key]
	if !ok {
		return nil, false
	}
	var arr []json.RawMessage
	if err := json.Unmarshal(raw, &arr); err != nil {
		return nil, false
	}
	return arr, true
}

// rawMap extracts an object field from the raw body map.
func rawMap(body map[string]json.RawMessage, key string) (map[string]json.RawMessage, bool) {
	raw, ok := body[key]
	if !ok {
		return nil, false
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(raw, &m); err != nil {
		return nil, false
	}
	return m, true
}

// ---------------------------------------------------------------------------
// Domain request builders
// ---------------------------------------------------------------------------

// buildInvokeRequest assembles a domain.InvokeRequest from the HTTP identity
// + the OpenAI body. surface is the ADR-252 surface token ("openai_compat"
// for chat/images, "responses" for the responses endpoint).
func (s *Server) buildInvokeRequest(r *http.Request, id identity, model string, body map[string]json.RawMessage, surface string) domain.InvokeRequest {
	req := domain.InvokeRequest{
		TenantID:         id.tenantID,
		GCID:             id.gcid,
		AgentID:          id.agentID,
		LogicalModelID:   domain.LogicalModelID(model),
		Traceparent:      id.traceparent,
		Tracestate:       id.tracestate,
		Surface:          surface,
		ActionCode:       rawString(body, "user"),
	}

	// Generation config: temperature / top_p / stop / n / seed / max_tokens.
	if genCfg := generationConfig(body); len(genCfg) > 0 {
		req.GenerationConfig = genCfg
	}

	return req
}

// generationConfig extracts the vendor-neutral generation parameters from
// the OpenAI body, mirroring the Python compat.generation_params.
func generationConfig(body map[string]json.RawMessage) map[string]any {
	out := map[string]any{}
	for _, key := range []string{"temperature", "top_p", "stop", "n", "seed"} {
		if raw, ok := body[key]; ok {
			var v any
			if err := json.Unmarshal(raw, &v); err == nil && v != nil {
				out[key] = v
			}
		}
	}
	// max_tokens falls back to max_completion_tokens.
	maxRaw, ok := body["max_tokens"]
	if !ok {
		maxRaw, ok = body["max_completion_tokens"]
	}
	if ok {
		var v any
		if err := json.Unmarshal(maxRaw, &v); err == nil && v != nil {
			out["max_tokens"] = v
		}
	}
	return out
}

// ---------------------------------------------------------------------------
// Error mapping — domain errors → OpenAI error envelope
// ---------------------------------------------------------------------------

// mappedError is the result of mapping a domain error (or in-band finish
// reason) to an OpenAI HTTP status + error envelope.
type mappedError struct {
	status  int
	errType string
	code    string
	message string
}

// mapInvokeError translates a domain.InvokeError (or PreconditionError) into
// the OpenAI error envelope. The mapping mirrors the Python facade's
// exception handlers:
//
//   - An upstream HTTP status is relayed: 401/403 → authentication_error,
//     429 → rate_limit_error, 404 → invalid_request_error, else →
//     upstream_error.
//   - A config error (Kind=config) → 500 server_error.
//   - A bare validation error (Kind=bare) → 400 invalid_request_error.
//   - An invoke error (Kind=invoke, the default) → 502 upstream_error.
//   - A PreconditionError → 400 surface_unstamped, 403 companion_suspended,
//     503 suspension_unreadable.
//
// ERROR SANITIZATION: the client-facing message NEVER includes the raw
// upstream error body. The upstream body may contain internal hostnames,
// stack traces, or other sensitive information. The gateway-controlled
// Detail field is used instead; the full error (with the upstream body) is
// available in the logs for internal debugging.
func mapInvokeError(err error) mappedError {
	// Per-tenant concurrency ceiling (CHORA_LLM_MAX_CONCURRENT_PER_TENANT): the
	// request never reached a provider and nothing was billed, so the OpenAI
	// envelope is a 429 rate_limit_error (retryable), not a 502 vendor error.
	var overloaded *domain.OverloadedError
	if errors.As(err, &overloaded) {
		return mappedError{http.StatusTooManyRequests, "rate_limit_error", "tenant_concurrency_limit", overloaded.Error()}
	}
	// PreconditionError — the companion-suspension decorator's typed refusal.
	var perr *domain.PreconditionError
	if errors.As(err, &perr) {
		switch perr.Reason {
		case domain.DenySurfaceUnstamped:
			return mappedError{http.StatusBadRequest, "invalid_request_error", "surface_unstamped", perr.Detail}
		case domain.DenyCompanionSuspended:
			return mappedError{http.StatusForbidden, "insufficient_quota", "companion_suspended", perr.Detail}
		case domain.DenySuspensionUnreadable:
			return mappedError{http.StatusServiceUnavailable, "upstream_error", "suspension_unreadable", perr.Detail}
		default:
			return mappedError{http.StatusForbidden, "invalid_request_error", perr.Reason, perr.Detail}
		}
	}

	var ierr *domain.InvokeError
	if !errors.As(err, &ierr) {
		return mappedError{http.StatusBadGateway, "upstream_error", "internal_error", "internal error"}
	}

	// Upstream HTTP status relay — the provider's own status is relayed
	// rather than flattened, so a caller can distinguish an auth rejection
	// from a rate limit from a provider outage. The message is sanitized:
	// the upstream error body is NOT included.
	if ierr.UpstreamStatus != nil {
		return mappedError{
			status:  *ierr.UpstreamStatus,
			errType: upstreamErrorType(*ierr.UpstreamStatus),
			code:    "upstream_error",
			message: domain.SanitizeUpstreamError(*ierr.UpstreamStatus, ""),
		}
	}

	switch ierr.Kind {
	case domain.ErrorKindConfig:
		return mappedError{http.StatusInternalServerError, "server_error", "gateway_misconfigured", ierr.Detail}
	case domain.ErrorKindBare:
		return mappedError{http.StatusBadRequest, "invalid_request_error", "invalid_request", ierr.Detail}
	default:
		return mappedError{http.StatusBadGateway, "upstream_error", "vendor_error", ierr.Detail}
	}
}

// upstreamErrorType maps an upstream HTTP status to the OpenAI error type,
// mirroring the Python facade's _upstream_error_type.
func upstreamErrorType(status int) string {
	switch {
	case status == 401 || status == 403:
		return "authentication_error"
	case status == 429:
		return "rate_limit_error"
	case status == 404:
		return "invalid_request_error"
	default:
		return "upstream_error"
	}
}

// mapInbandFinish maps an in-band domain.InvokeResponse finish reason to an
// OpenAI error envelope. Returns ok=false when the finish reason is a
// normal completion (no error). Budget and mana blocks → 402
// insufficient_quota; Armor blocks → 403.
func mapInbandFinish(resp domain.InvokeResponse) (mappedError, bool) {
	switch resp.FinishReason {
	case domain.FinishReasonBudgetBlock:
		return mappedError{
			status:  http.StatusPaymentRequired,
			errType: "insufficient_quota",
			code:    "budget_exhausted",
			message: resp.FinishDetail,
		}, true
	case domain.FinishReasonManaBlock:
		return mappedError{
			status:  http.StatusPaymentRequired,
			errType: "insufficient_quota",
			code:    "insufficient_mana",
			message: resp.FinishDetail,
		}, true
	case domain.FinishReasonModelArmorBlock:
		return mappedError{
			status:  http.StatusForbidden,
			errType: "insufficient_quota",
			code:    "model_armor_block",
			message: resp.FinishDetail,
		}, true
	default:
		return mappedError{}, false
	}
}

// ---------------------------------------------------------------------------
// Response translation — domain responses → OpenAI format
// ---------------------------------------------------------------------------

// chatCompletionResponse renders a domain.InvokeResponse as an OpenAI
// chat.completion object, mirroring the Python facade's _chat_response.
func (s *Server) chatCompletionResponse(resp domain.InvokeResponse, grounded bool) map[string]any {
	message := map[string]any{
		"role":    "assistant",
		"content": resp.Completion,
	}
	if resp.ToolCallsJSON != "" {
		var toolCalls []map[string]any
		if err := json.Unmarshal([]byte(resp.ToolCallsJSON), &toolCalls); err == nil {
			message["tool_calls"] = toolCalls
		}
	}

	meta := map[string]any{
		"vendor":        resp.Vendor,
		"latency_ms":    resp.LatencyMs,
		"invocation_id": resp.InvocationID,
	}
	if len(resp.FallbackChain) > 0 {
		meta["fallback_chain"] = resp.FallbackChain
	}
	if grounded {
		meta["grounded"] = true
	}
	if len(resp.Citations) > 0 {
		meta["citations"] = citationsToJSON(resp.Citations)
	}
	if len(resp.SearchQueries) > 0 {
		meta["search_queries"] = resp.SearchQueries
	}

	return map[string]any{
		"id":      resp.InvocationID,
		"object":  "chat.completion",
		"created": s.now().Unix(),
		"model":   resp.ModelVersion,
		"choices": []map[string]any{
			{
				"index":         0,
				"message":       message,
				"finish_reason": openaiFinishReason(resp.FinishReason),
			},
		},
		"usage": map[string]any{
			"prompt_tokens":     resp.Usage.InputTokens,
			"completion_tokens": resp.Usage.OutputTokens,
			"total_tokens":      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
		"chora_gateway": meta,
	}
}

// openaiFinishReason maps a domain FinishReason to the OpenAI finish_reason
// token. The old facade maps every non-length finish reason to "stop".
func openaiFinishReason(fr domain.FinishReason) string {
	if fr == domain.FinishReasonMaxTokens {
		return "length"
	}
	return "stop"
}

// responsesResponse renders a domain.InvokeResponse as an OpenAI Responses
// API object, mirroring the Python facade's _responses_response.
func (s *Server) responsesResponse(resp domain.InvokeResponse, grounded bool) map[string]any {
	output := []map[string]any{}
	if grounded {
		action := map[string]any{"type": "search"}
		if len(resp.SearchQueries) == 1 {
			action["query"] = resp.SearchQueries[0]
		} else if len(resp.SearchQueries) > 1 {
			action["queries"] = resp.SearchQueries
		}
		output = append(output, map[string]any{
			"type":   "web_search_call",
			"id":     "ws_" + resp.InvocationID,
			"status": "completed",
			"action": action,
		})
	}

	// The Go domain carries a single Completion + Citations (no multi-message
	// list), so the responses output is one message item.
	output = append(output, map[string]any{
		"type":   "message",
		"id":     "msg_" + resp.InvocationID,
		"status": "completed",
		"role":   "assistant",
		"content": []map[string]any{
			{
				"type":        "output_text",
				"text":        resp.Completion,
				"annotations": citationsToJSON(resp.Citations),
			},
		},
	})

	meta := map[string]any{
		"vendor":        resp.Vendor,
		"grounded":      grounded,
		"citations":     citationsToJSON(resp.Citations),
		"latency_ms":    resp.LatencyMs,
		"invocation_id": resp.InvocationID,
	}
	if len(resp.SearchQueries) > 0 {
		meta["search_queries"] = resp.SearchQueries
	}
	if len(resp.FallbackChain) > 0 {
		meta["fallback_chain"] = resp.FallbackChain
	}
	if resp.FinishDetail != "" {
		meta["finish_reason"] = resp.FinishReason.String()
		meta["finish_detail"] = resp.FinishDetail
	}

	return map[string]any{
		"id":         resp.InvocationID,
		"object":     "response",
		"created_at": s.now().Unix(),
		"model":      resp.ModelVersion,
		"status":     "completed",
		"output":     output,
		"usage": map[string]any{
			"prompt_tokens":     resp.Usage.InputTokens,
			"completion_tokens": resp.Usage.OutputTokens,
			"total_tokens":      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
		"chora_gateway": meta,
	}
}

// imageResponse renders a domain.InvokeResponse as an OpenAI
// images/generations response, mirroring the Python facade's image handler.
func (s *Server) imageResponse(resp domain.InvokeResponse) map[string]any {
	out := map[string]any{
		"created": s.now().Unix(),
		"data":    []map[string]any{{"b64_json": base64.StdEncoding.EncodeToString(resp.ImageBytes)}},
		"usage": map[string]any{
			"prompt_tokens":     resp.Usage.InputTokens,
			"completion_tokens": resp.Usage.OutputTokens,
			"total_tokens":      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	}
	if resp.ImageMIMEType != "" {
		out["mime_type"] = resp.ImageMIMEType
	}
	return out
}

// citationsToJSON converts domain citations to the OpenAI url_citation
// annotation shape, mirroring the Python facade's _annotations_to_json.
func citationsToJSON(citations []domain.Citation) []map[string]any {
	out := make([]map[string]any, 0, len(citations))
	for _, c := range citations {
		annotation := map[string]any{"type": "url_citation", "url": c.URL}
		if c.Title != "" {
			annotation["title"] = c.Title
		}
		if c.StartIndex != 0 || c.EndIndex != 0 {
			annotation["start_index"] = c.StartIndex
			annotation["end_index"] = c.EndIndex
		}
		out = append(out, annotation)
	}
	return out
}

// ---------------------------------------------------------------------------
// Grounding surface check
// ---------------------------------------------------------------------------

// groundingSurfaceFor resolves the effective grounding surface for a model,
// returning (surface, error). Mirrors the Python facade's
// grounding_surface_for: the model must exist, advertise web_search, and
// have a grounding block configured.
func (s *Server) groundingSurfaceFor(model string) (string, mappedError, bool) {
	spec, ok := s.registry.Get(model)
	if !ok {
		return "", mappedError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			code:    "grounding_unavailable",
			message: fmt.Sprintf("model \"%s\" is not in the registry", model),
		}, false
	}
	if !spec.Supports("web_search") {
		return "", mappedError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			code:    "grounding_unavailable",
			message: fmt.Sprintf("model \"%s\" does not advertise the \"web_search\" capability (it has: %s)",
				model, strings.Join(spec.Capabilities, ", ")),
		}, false
	}
	if spec.Grounding == nil {
		return "", mappedError{
			status:  http.StatusBadRequest,
			errType: "invalid_request_error",
			code:    "grounding_unavailable",
			message: fmt.Sprintf("model \"%s\" advertises \"web_search\" but its registry entry configures no grounding endpoint", model),
		}, false
	}
	return spec.Grounding.EffectiveSurface(spec.Provider), mappedError{}, true
}

// wantsGrounding reports whether the tool list contains a web_search or
// web_fetch tool, mirroring the Python compat.wants_grounding.
func wantsGrounding(tools []json.RawMessage) bool {
	for _, raw := range tools {
		var tool map[string]any
		if err := json.Unmarshal(raw, &tool); err != nil {
			continue
		}
		for _, key := range []string{"type", "name"} {
			value, _ := tool[key].(string)
			lower := strings.ToLower(value)
			if strings.Contains(lower, "web_search") || strings.Contains(lower, "web_fetch") {
				return true
			}
		}
	}
	return false
}
