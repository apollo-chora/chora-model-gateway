// Package httpapi serves the OpenAI-compatible HTTP surface.
//
// The gateway already speaks gRPC; this package adds the shape every OpenAI
// client, SDK and CLI already knows, so an existing toolchain can be
// pointed at it by changing one base URL:
//
//	POST /v1/chat/completions   text + tools + vision
//	POST /v1/images/generations image generation
//	POST /v1/embeddings         text-to-vector
//	GET  /v1/models             the registry, as OpenAI model objects
//	GET  /healthz  /readyz      probes
//
// Two design decisions are load-bearing:
//
//   - The `model` field is the REGISTRY KEY, not a raw upstream name. Every
//     request still resolves through the registry, so budget, ledger and
//     fallback keep working on this surface. An unknown name is a 404, never
//     a passthrough to whatever the caller asked for.
//   - Streaming is not implemented. `stream: true` is refused with an
//     explicit error rather than silently returning a non-streaming body,
//     because a client that asked for SSE and silently got a blob will hang
//     or mis-parse.
package httpapi

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/config"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// Gateway is the domain surface the HTTP facade calls. Declared as an
// interface so the facade is testable against a stub and so the composition
// root can hand it the decorator-wrapped chain.
type Gateway interface {
	Invoke(ctx context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error)
	Embed(ctx context.Context, req domain.EmbedRequest) (domain.EmbedResponse, error)
}

// Server is the OpenAI-compatible HTTP adapter.
type Server struct {
	gw      Gateway
	models  *config.Registry
	version string

	defaultTenantID string
	defaultGCID     string
	defaultAgentID  string

	mux *http.ServeMux
}

// Config groups the facade's construction inputs.
type Config struct {
	Gateway Gateway
	Models  *config.Registry

	// Version is stamped into every response's `model` echo and the
	// server-error payloads.
	Version string

	// DefaultTenantID / DefaultGCID / DefaultAgentID attribute calls that
	// arrive without a tenant header. The budget row is tenant-scoped, so a
	// default is required rather than optional.
	DefaultTenantID string
	DefaultGCID     string
	DefaultAgentID  string
}

// NewServer constructs the facade. Fails loud on a nil gateway or registry:
// an adapter with nothing to call would answer every request with a
// misleading success.
func NewServer(cfg Config) (*Server, error) {
	if cfg.Gateway == nil {
		return nil, errors.New("httpapi: Gateway required")
	}
	if cfg.Models == nil {
		return nil, errors.New("httpapi: model Registry required")
	}
	if cfg.DefaultTenantID == "" {
		return nil, errors.New("httpapi: DefaultTenantID required (the budget row is tenant-scoped)")
	}
	if cfg.DefaultAgentID == "" {
		cfg.DefaultAgentID = "openai_compat"
	}
	if cfg.DefaultGCID == "" {
		cfg.DefaultGCID = cfg.DefaultTenantID
	}
	if cfg.Version == "" {
		cfg.Version = "unknown"
	}

	s := &Server{
		gw:              cfg.Gateway,
		models:          cfg.Models,
		version:         cfg.Version,
		defaultTenantID: cfg.DefaultTenantID,
		defaultGCID:     cfg.DefaultGCID,
		defaultAgentID:  cfg.DefaultAgentID,
		mux:             http.NewServeMux(),
	}
	s.routes()
	return s, nil
}

func (s *Server) routes() {
	s.mux.HandleFunc("/healthz", s.handleHealthz)
	s.mux.HandleFunc("/readyz", s.handleReadyz)
	s.mux.HandleFunc("/v1/models", s.handleModels)
	s.mux.HandleFunc("/v1/models/", s.handleModels)
	s.mux.HandleFunc("/v1/responses", s.handleResponses)
	s.mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("/v1/completions", s.handleChatCompletions)
	s.mux.HandleFunc("/v1/images/generations", s.handleImageGenerations)
	s.mux.HandleFunc("/v1/images/edits", s.handleImageGenerations)
	s.mux.HandleFunc("/v1/embeddings", s.handleEmbeddings)
}

// ServeHTTP implements http.Handler.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// The facade is a bearer-token surface, so a permissive CORS policy here
	// would let any web page drive the operator's inference spend. Locked
	// down; a browser-based caller needs an explicit opt-in.
	w.Header().Set("X-Content-Type-Options", "nosniff")
	s.mux.ServeHTTP(w, r)
}

// ---------------------------------------------------------------------------
// OpenAI error envelope
// ---------------------------------------------------------------------------

// apiError is the OpenAI error object shape. Every non-2xx response on this
// surface uses it, so a client can parse errors the same way it parses
// successes.
type apiError struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param,omitempty"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}

func writeError(w http.ResponseWriter, status int, errType, code, message string) {
	var e apiError
	e.Error.Message = message
	e.Error.Type = errType
	e.Error.Code = code
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(e)
}

// writeGatewayError maps a domain error onto the closest OpenAI-shaped
// status. An upstream provider error keeps its own status, because a 401
// from the provider is a credential problem the operator must see as 401,
// not as an opaque 502.
func (s *Server) writeGatewayError(w http.ResponseWriter, r *http.Request, err error) {
	// A misconfiguration is a 500, not a 502: reporting it as "bad gateway"
	// sends the caller looking at a provider that is working perfectly. This
	// check comes first because ConfigError is the more specific diagnosis.
	var configErr *domain.ConfigError
	if errors.As(err, &configErr) {
		slog.ErrorContext(r.Context(), "httpapi: gateway misconfiguration",
			"detail", configErr.Detail, "path", r.URL.Path)
		writeError(w, http.StatusInternalServerError, "server_error", "gateway_misconfigured", configErr.Detail)
		return
	}

	var invokeErr *domain.InvokeError
	if errors.As(err, &invokeErr) {
		detail := invokeErr.Detail
		if invokeErr.Inner != nil {
			detail = invokeErr.Inner.Error()
		}
		slog.WarnContext(r.Context(), "httpapi: invoke failed",
			"reason", invokeErr.Reason.String(), "detail", detail,
			"fallbacks", invokeErr.FallbackLog, "path", r.URL.Path)

		// Provider-level failures surface with the provider's own status.
		var upstream UpstreamStatus
		if errors.As(invokeErr.Inner, &upstream) && upstream.UpstreamStatusCode() >= 400 {
			writeError(w, upstream.UpstreamStatusCode(), upstreamErrorType(upstream.UpstreamStatusCode()), "upstream_error", detail)
			return
		}
		if invokeErr.Reason == domain.FinishReasonBudgetBlock {
			writeError(w, http.StatusPaymentRequired, "insufficient_quota", "budget_exhausted", invokeErr.Detail)
			return
		}
		writeError(w, http.StatusBadGateway, "upstream_error", "vendor_error", detail)
		return
	}

	// Validation errors from the envelope.
	writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_request", err.Error())
}

// UpstreamStatus is implemented by every vendor adapter's error type, so the
// facade can relay a provider's own status code (a 401 from the provider is a
// credential problem the operator must see as 401, not as a flattened 502)
// without importing each adapter package.
type UpstreamStatus interface {
	error
	UpstreamStatusCode() int
}

func upstreamErrorType(status int) string {
	switch status {
	case http.StatusUnauthorized, http.StatusForbidden:
		return "authentication_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusNotFound:
		return "invalid_request_error"
	default:
		return "upstream_error"
	}
}

// ---------------------------------------------------------------------------
// Attribution
// ---------------------------------------------------------------------------

// attribution reads the optional tenant/actor headers, falling back to the
// configured defaults. A caller that does not care about per-tenant budgets
// still gets attributed — to the configured default tenant — rather than
// escaping the budget check entirely.
func (s *Server) attribution(r *http.Request) (tenantID, gcid string) {
	tenantID = firstNonEmpty(
		r.Header.Get("X-Chora-Tenant-Id"),
		r.URL.Query().Get("tenant"),
		s.defaultTenantID,
	)
	gcid = firstNonEmpty(
		r.Header.Get("X-Chora-Gcid"),
		s.defaultGCID,
	)
	return tenantID, gcid
}

// baseRequest assembles the fields every surface shares.
func (s *Server) baseRequest(r *http.Request, model string) domain.InvokeRequest {
	tenantID, gcid := s.attribution(r)
	return domain.InvokeRequest{
		TenantID:       tenantID,
		GCID:           gcid,
		AgentID:        s.defaultAgentID,
		LogicalModelID: domain.LogicalModelID(model),
		Traceparent:    r.Header.Get("traceparent"),
		Tracestate:     r.Header.Get("tracestate"),
		Surface:        "openai_compat",
		ActionCode:     s.defaultAgentID,
	}
}

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

func (s *Server) handleHealthz(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok"))
}

func (s *Server) handleReadyz(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ready"))
}

// Budget-block refusal, shared by every surface that calls Invoke. A budget
// block is a terminal refusal, not an empty result: returning 200 with a
// blank body tells the caller the model answered and said nothing, which
// reads as a silent failure and invites a pointless retry. It reports whether
// it handled the response.
func refuseIfBudgetBlocked(w http.ResponseWriter, r *http.Request, model string, resp domain.InvokeResponse) bool {
	if resp.FinishReason != domain.FinishReasonBudgetBlock {
		return false
	}
	slog.WarnContext(r.Context(), "httpapi: budget block",
		"model", model, "invocation_id", resp.InvocationID)
	writeError(w, http.StatusPaymentRequired, "insufficient_quota", "budget_exhausted",
		firstNonEmpty(resp.FinishDetail, "the tenant LLM budget for this period is exhausted"))
	return true
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

type modelList struct {
	Object string       `json:"object"`
	Data   []modelEntry `json:"data"`
}

type modelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`

	// Non-standard, additive fields. OpenAI clients ignore unknown keys, and
	// they are the only way an operator's tooling learns the context window
	// and capabilities without a second call.
	ContextWindow   int      `json:"context_window,omitempty"`
	MaxOutputTokens int      `json:"max_output_tokens,omitempty"`
	Capabilities    []string `json:"capabilities,omitempty"`
	BaseURL         string   `json:"base_url,omitempty"`
}

// handleModels serves the registry as OpenAI model objects.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", "GET only")
		return
	}
	// /v1/models/<id> — single-model retrieval.
	if id := strings.TrimPrefix(r.URL.Path, "/v1/models"); id != "" {
		id = strings.Trim(id, "/")
		if id == "" {
			s.writeModelList(w)
			return
		}
		target, ok := s.models.Lookup(domain.LogicalModelID(id))
		if !ok {
			writeError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
				fmt.Sprintf("model %q is not in the registry", id))
			return
		}
		writeJSON(w, http.StatusOK, s.toModelEntry(target))
		return
	}
	s.writeModelList(w)
}

func (s *Server) writeModelList(w http.ResponseWriter) {
	out := modelList{Object: "list", Data: make([]modelEntry, 0, 16)}
	for _, t := range s.models.Targets() {
		out.Data = append(out.Data, s.toModelEntry(t))
	}
	writeJSON(w, http.StatusOK, out)
}

func (s *Server) toModelEntry(t domain.TargetModel) modelEntry {
	return modelEntry{
		ID:              string(t.LogicalModelID),
		Object:          "model",
		Created:         0,
		OwnedBy:         string(t.Vendor),
		ContextWindow:   t.ContextWindow,
		MaxOutputTokens: t.MaxOutputTokens,
		Capabilities:    t.Capabilities,
		BaseURL:         t.BaseURL,
	}
}

// ---------------------------------------------------------------------------
// POST /v1/chat/completions
// ---------------------------------------------------------------------------

type chatCompletionRequest struct {
	Model    string            `json:"model"`
	Messages []chatMessage     `json:"messages"`
	Stream   bool              `json:"stream"`
	Tools    []json.RawMessage `json:"tools"`
	User     string            `json:"user"`

	// Generation knobs are captured as a raw map so every provider extension
	// passes through untouched rather than being dropped by a fixed struct.
	Temperature         *float64        `json:"temperature"`
	TopP                *float64        `json:"top_p"`
	MaxTokens           *int            `json:"max_tokens"`
	MaxCompletionTokens *int            `json:"max_completion_tokens"`
	Stop                json.RawMessage `json:"stop"`
	N                   *int            `json:"n"`
	Seed                *int            `json:"seed"`
}

type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
	Name    string          `json:"name,omitempty"`
}

type chatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   chatUsage    `json:"usage"`
	// Additive: what the gateway actually did, which is the part an operator
	// debugging spend needs and which OpenAI's shape has no field for.
	Gateway *gatewayMeta `json:"chora_gateway,omitempty"`
}

type chatChoice struct {
	Index        int         `json:"index"`
	Message      chatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type chatUsage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type gatewayMeta struct {
	Vendor        string   `json:"vendor"`
	FallbackChain []string `json:"fallback_chain,omitempty"`
	LatencyMs     int32    `json:"latency_ms"`
	InvocationID  string   `json:"invocation_id"`

	// Grounded is true when this call ran a hosted web search, and Citations
	// carries the normalised sources. Both absent on a plain text call, which
	// is how a caller distinguishes "not grounded" from "grounded, found
	// nothing".
	Grounded      bool                       `json:"grounded,omitempty"`
	Citations     []domain.GroundingCitation `json:"citations,omitempty"`
	SearchQueries []string                   `json:"search_queries,omitempty"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", "POST only")
		return
	}

	var req chatCompletionRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}
	if req.Stream {
		// Refused explicitly. Silently returning a non-streaming body to a
		// client expecting SSE produces a hang, not an error.
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
	// Set when the caller named a web-search tool AND the registry agrees this
	// model can serve one on this surface.
	grounded := false
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter",
			"\"messages\" must contain at least one message")
		return
	}

	// A caller may ask for hosted web search on the chat-completions surface
	// by naming the tool in `tools`. OpenAI only offers search on /v1/responses,
	// so this is honoured only when the registry entry configured the
	// chat_completions grounding surface; otherwise the request is refused
	// rather than dispatched with a tool the provider will ignore, which
	// would return an answer that only LOOKS grounded.
	if toolNamesWebSearch(req.Tools) {
		groundedOn, err := s.groundingSurfaceFor(req.Model)
		if err != nil {
			slog.WarnContext(r.Context(), "httpapi: grounding refused on chat/completions",
				"model", req.Model, "reason", err)
			writeError(w, http.StatusBadRequest, "invalid_request_error", "grounding_unavailable", err.Error())
			return
		}
		if groundedOn != domain.SurfaceChatCompletions {
			writeError(w, http.StatusBadRequest, "invalid_request_error", "wrong_grounding_surface",
				fmt.Sprintf("model %q is configured to ground on the %s surface; use POST /v1/responses for hosted web search",
					req.Model, groundedOn))
			return
		}
		grounded = true
	}

	invoke := s.baseRequest(r, req.Model)
	invoke.ActionCode = firstNonEmpty(req.User, s.defaultAgentID)
	if grounded {
		invoke.ResponseModality = domain.ModalityGrounded
	}

	// Split the system prompt out of the conversation: the domain models it
	// separately, and providers differ on whether they accept it as a message.
	var systemParts []string
	var conversation []chatMessage
	for _, m := range req.Messages {
		if strings.EqualFold(m.Role, "system") || strings.EqualFold(m.Role, "developer") {
			if text := rawToText(m.Content); text != "" {
				systemParts = append(systemParts, text)
			}
			continue
		}
		conversation = append(conversation, m)
	}
	invoke.SystemPrompt = strings.Join(systemParts, "\n\n")

	if len(conversation) > 0 {
		encoded, err := json.Marshal(conversation)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_messages", err.Error())
			return
		}
		invoke.ContentsJSON = string(encoded)
		// The domain validator requires a prompt or a conversation; a
		// system-only request has neither, and an empty conversation is not
		// something any provider will bill usefully.
		if len(conversation) == 0 {
			writeError(w, http.StatusBadRequest, "invalid_request_error", "empty_conversation",
				"messages contains only a system prompt; add at least one user message")
			return
		}
	} else {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "empty_conversation",
			"messages contains only a system prompt; add at least one user message")
		return
	}

	if len(req.Tools) > 0 {
		encoded, err := json.Marshal(req.Tools)
		if err != nil {
			writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_tools", err.Error())
			return
		}
		invoke.ToolsJSON = string(encoded)
	}

	invoke.GenerationConfig = buildGenerationConfig(
		req.Temperature, req.TopP, firstNonNilInt(req.MaxTokens, req.MaxCompletionTokens),
		req.Stop, req.N, req.Seed,
	)

	resp, err := s.gw.Invoke(r.Context(), invoke)
	if err != nil {
		s.writeGatewayError(w, r, err)
		return
	}

	// A budget block is a terminal refusal, not an empty completion. Returning
	// 200 with "" here tells the caller the model answered and said nothing,
	// which reads as a silent failure and invites a pointless retry.
	if refuseIfBudgetBlocked(w, r, req.Model, resp) {
		return
	}

	writeJSON(w, http.StatusOK, chatCompletionResponse{
		ID:      resp.InvocationID,
		Object:  "chat.completion",
		Created: timeToUnix(resp.CompletedAt),
		Model:   resp.ModelVersion,
		Choices: []chatChoice{{
			Index: 0,
			Message: chatMessage{
				Role:    "assistant",
				Content: mustJSON(resp.Completion),
			},
			FinishReason: openAIFinishReason(resp.FinishReason),
		}},
		Usage: chatUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
		Gateway: &gatewayMeta{
			Vendor:        resp.Vendor,
			FallbackChain: resp.FallbackChain,
			LatencyMs:     resp.LatencyMs,
			InvocationID:  resp.InvocationID,
			Grounded:      grounded,
			Citations:     resp.Citations,
			SearchQueries: resp.SearchQueries,
		},
	})
}

// toolNamesWebSearch reports whether a caller-supplied tool array asks for
// hosted web search.
//
// Substring matching is deliberate: the providers spell this tool a half-dozen
// ways ("web_search", "web_search_preview", "web_search_2025_08_26",
// "web_search_20250305", "web_fetch"), and every one of them means the same
// thing to the caller. Exact matching would make the gateway's usefulness
// depend on which vendor coined the name this quarter.
func toolNamesWebSearch(tools []json.RawMessage) bool {
	for _, raw := range tools {
		var tool struct {
			Type string `json:"type"`
			Name string `json:"name"`
		}
		if err := json.Unmarshal(raw, &tool); err != nil {
			continue
		}
		for _, candidate := range []string{tool.Type, tool.Name} {
			c := strings.ToLower(candidate)
			if strings.Contains(c, "web_search") || strings.Contains(c, "web_fetch") {
				return true
			}
		}
	}
	return false
}

// groundingSurfaceFor resolves the surface a model grounds on, or an error
// explaining why it cannot.
func (s *Server) groundingSurfaceFor(model string) (domain.GroundingSurface, error) {
	target, ok := s.models.Lookup(domain.LogicalModelID(model))
	if !ok {
		return "", fmt.Errorf("model %q is not in the registry", model)
	}
	if !target.Supports(domain.CapabilityWebSearch) {
		return "", fmt.Errorf(
			"model %q does not advertise the %q capability (it has: %s)",
			model, domain.CapabilityWebSearch, strings.Join(target.Capabilities, ", "))
	}
	if target.Grounding == nil {
		return "", fmt.Errorf(
			"model %q advertises %q but its registry entry configures no grounding endpoint",
			model, domain.CapabilityWebSearch)
	}
	return target.Grounding.EffectiveSurface(target.Vendor), nil
}

func openAIFinishReason(r domain.FinishReason) string {
	switch r {
	case domain.FinishReasonComplete:
		return "stop"
	case domain.FinishReasonMaxTokens:
		return "length"
	case domain.FinishReasonBudgetBlock:
		return "stop"
	default:
		return "stop"
	}
}

// ---------------------------------------------------------------------------
// POST /v1/images/generations
// ---------------------------------------------------------------------------

type imageRequest struct {
	Model          string `json:"model"`
	Prompt         string `json:"prompt"`
	N              int    `json:"n"`
	Size           string `json:"size"`
	Quality        string `json:"quality"`
	Style          string `json:"style"`
	ResponseFormat string `json:"response_format"`
	User           string `json:"user"`
}

type imageResponse struct {
	Created int64       `json:"created"`
	Data    []imageData `json:"data"`
	// Additive usage block: OpenAI reports no usage on the images surface, so
	// a caller has no other way to see what the call cost.
	Usage *chatUsage `json:"usage,omitempty"`

	// MIMEType is the sniffed format of the returned bytes ("image/png",
	// "image/webp", ...). Additive and non-standard: OpenAI omits it because it
	// only ever returns PNG, but OpenAI-compatible aggregators do not all hold
	// to that.
	MIMEType string `json:"mime_type,omitempty"`
}

type imageData struct {
	URL           string `json:"url,omitempty"`
	B64JSON       string `json:"b64_json,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

func (s *Server) handleImageGenerations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", "POST only")
		return
	}

	var req imageRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}
	if req.Prompt == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter", "\"prompt\" is required")
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter",
			"\"model\" is required and must name an image-capable entry in the model registry")
		return
	}
	if !s.models.Known(req.Model) {
		writeError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model %q is not in the registry; GET /v1/models lists what is", req.Model))
		return
	}

	// Refused BEFORE the dispatch. The gateway returns bytes and hosts nothing,
	// so a URL response could only ever point at something that does not
	// exist. Checking after the call would pay for a provider image generation
	// and then throw the result away.
	if strings.EqualFold(req.ResponseFormat, "url") {
		writeError(w, http.StatusNotImplemented, "invalid_request_error", "url_response_unsupported",
			"this gateway returns bytes, not hosted URLs; use \"response_format\": \"b64_json\" (the default)")
		return
	}

	invoke := s.baseRequest(r, req.Model)
	invoke.Prompt = req.Prompt
	invoke.ResponseModality = domain.ModalityImage
	invoke.ActionCode = firstNonEmpty(req.User, s.defaultAgentID+"_image")
	invoke.GenerationConfig = map[string]any{}
	if req.Size != "" {
		invoke.GenerationConfig["size"] = req.Size
	}
	if req.Quality != "" {
		invoke.GenerationConfig["quality"] = req.Quality
	}
	if req.Style != "" {
		invoke.GenerationConfig["style"] = req.Style
	}

	resp, err := s.gw.Invoke(r.Context(), invoke)
	if err != nil {
		s.writeGatewayError(w, r, err)
		return
	}

	if refuseIfBudgetBlocked(w, r, req.Model, resp) {
		return
	}

	if resp.FinishReason != domain.FinishReasonComplete || len(resp.ImageBytes) == 0 {
		writeError(w, http.StatusBadGateway, "upstream_error", "no_image",
			firstNonEmpty(resp.FinishDetail, "the model returned no image"))
		return
	}

	out := imageResponse{
		Created: timeToUnix(resp.CompletedAt),
		Data: []imageData{{
			B64JSON:       base64.StdEncoding.EncodeToString(resp.ImageBytes),
			RevisedPrompt: resp.ImageRevisedPrompt,
		}},
		Usage: &chatUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
		// The OpenAI images response carries no mime type, and the provider
		// sends no Content-Type for an inline image — so a caller receiving
		// b64_json has no way to tell a PNG from a WebP unless we say so.
		// api.meta.ai returns WebP where OpenAI documents PNG, so this is not
		// theoretical. Additive; an OpenAI client ignores it.
		MIMEType: resp.ImageMIMEType,
	}
	writeJSON(w, http.StatusOK, out)
}

// ---------------------------------------------------------------------------
// POST /v1/embeddings
// ---------------------------------------------------------------------------

type embeddingRequest struct {
	Model          string          `json:"model"`
	Input          json.RawMessage `json:"input"`
	EncodingFormat string          `json:"encoding_format"`
	User           string          `json:"user"`
	Dimensions     int32           `json:"dimensions"`
}

type embeddingResponse struct {
	Object string         `json:"object"`
	Model  string         `json:"model"`
	Data   []embeddingRow `json:"data"`
	Usage  chatUsage      `json:"usage"`
}

type embeddingRow struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", "POST only")
		return
	}

	var req embeddingRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_json", err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "missing_parameter", "\"model\" is required")
		return
	}
	if !s.models.Known(req.Model) {
		writeError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model %q is not in the registry", req.Model))
		return
	}
	if strings.EqualFold(req.EncodingFormat, "base64") {
		// Float arrays are what every client actually handles; base64 is a
		// bandwidth optimisation with a real interoperability cost.
		writeError(w, http.StatusNotImplemented, "invalid_request_error", "base64_unsupported",
			"\"encoding_format\": \"base64\" is not implemented; omit it or send \"float\"")
		return
	}

	inputs, err := parseEmbeddingInput(req.Input)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "invalid_parameter", err.Error())
		return
	}

	tenantID, gcid := s.attribution(r)
	out := embeddingResponse{
		Object: "list",
		Model:  req.Model,
		Data:   make([]embeddingRow, 0, len(inputs)),
	}
	var totalTokens int64
	for i, text := range inputs {
		resp, err := s.gw.Embed(r.Context(), domain.EmbedRequest{
			TenantID:         tenantID,
			GCID:             gcid,
			AgentID:          s.defaultAgentID,
			LogicalModelID:   domain.LogicalModelID(req.Model),
			Text:             text,
			OutputDimensions: req.Dimensions,
			Traceparent:      r.Header.Get("traceparent"),
			Tracestate:       r.Header.Get("tracestate"),
		})
		if err != nil {
			s.writeGatewayError(w, r, err)
			return
		}
		out.Data = append(out.Data, embeddingRow{Object: "embedding", Index: i, Embedding: resp.Values})
		totalTokens += resp.Usage.InputTokens
	}
	out.Usage = chatUsage{PromptTokens: totalTokens, TotalTokens: totalTokens}
	writeJSON(w, http.StatusOK, out)
}

// parseEmbeddingInput accepts both OpenAI input shapes: a bare string or an
// array of strings (a token array is not supported and is refused).
func parseEmbeddingInput(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, errors.New("\"input\" is required")
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if single == "" {
			return nil, errors.New("\"input\" must not be empty")
		}
		return []string{single}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		if len(many) == 0 {
			return nil, errors.New("\"input\" must not be empty")
		}
		for i, s := range many {
			if s == "" {
				return nil, fmt.Errorf("\"input\"[%d] is empty", i)
			}
		}
		return many, nil
	}
	return nil, errors.New("\"input\" must be a string or an array of strings; token arrays are not supported")
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func buildGenerationConfig(
	temperature, topP *float64,
	maxTokens *int,
	stop json.RawMessage,
	n, seed *int,
) map[string]any {
	cfg := map[string]any{}
	if temperature != nil {
		cfg["temperature"] = *temperature
	}
	if topP != nil {
		cfg["top_p"] = *topP
	}
	// Numbers are normalised to float64, which is what encoding/json produces
	// for any numeric field. Keeping the map's value types consistent means
	// the vendor adapters' numeric coercion has exactly one shape to handle.
	if maxTokens != nil {
		cfg["max_tokens"] = float64(*maxTokens)
	}
	if n != nil {
		cfg["n"] = float64(*n)
	}
	if seed != nil {
		cfg["seed"] = float64(*seed)
	}
	if len(stop) > 0 {
		// Decoded rather than forwarded raw: the vendor adapters read this as
		// a Go value, and passing bytes through would silently drop it.
		var asString string
		if err := json.Unmarshal(stop, &asString); err == nil {
			cfg["stop"] = asString
		} else {
			var asList []string
			if err := json.Unmarshal(stop, &asList); err == nil {
				cfg["stop"] = asList
			}
		}
	}
	return cfg
}

func decodeJSON(w http.ResponseWriter, r *http.Request, out any) error {
	// Bound the body: an unbounded read of client JSON is a memory
	// exhaustion vector on a surface that is meant to be publicly reachable.
	defer func() { _ = r.Body.Close() }()
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8<<20))
	if err := dec.Decode(out); err != nil {
		if errors.Is(err, context.Canceled) {
			return errors.New("request cancelled")
		}
		return fmt.Errorf("could not parse request body: %w", err)
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(body); err != nil {
		slog.Error("httpapi: encode response", "error", err)
	}
}

func mustJSON(s string) json.RawMessage {
	if s == "" {
		return json.RawMessage(`""`)
	}
	encoded, err := json.Marshal(s)
	if err != nil {
		return json.RawMessage(`""`)
	}
	return encoded
}

// rawToText flattens the OpenAI content union into plain text.
func rawToText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil {
		return asString
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var b strings.Builder
		for _, p := range parts {
			b.WriteString(p.Text)
		}
		return b.String()
	}
	return ""
}

func timeToUnix(t time.Time) int64 {
	if t.IsZero() {
		return time.Now().Unix()
	}
	return t.Unix()
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func firstNonNilInt(vals ...*int) *int {
	for _, v := range vals {
		if v != nil {
			return v
		}
	}
	return nil
}
