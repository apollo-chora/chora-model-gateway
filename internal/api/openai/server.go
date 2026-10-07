// Package openai is the OpenAI-compatible HTTP API for the model gateway.
//
// The handlers are thin: they translate OpenAI-format requests into
// executor.ExecuteRequest calls on the Executor, and translate
// executor.ExecuteResponse back into OpenAI format. All governance (Model
// Armor, mana metering, companion suspension, budget) runs through the SAME
// Executor the gRPC adapter uses — no governance logic is duplicated here.
// Both gRPC and HTTP go through the exact same Execute() path.
//
// Routing uses net/http (stdlib) with Go 1.22+ pattern matching. Every
// response carries X-Content-Type-Options: nosniff; every route has a 405
// method guard; errors use the OpenAI envelope
// {"error": {"message": ..., "type": ..., "code": ...}}.
package openai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/executor"
	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/registry"
)

// Invoker is the subset of the Executor the HTTP handlers call. The Executor
// composes the governance pipeline (suspension → mana → domain service);
// *executor.Executor satisfies it. Both gRPC and HTTP adapters call the same
// Executor.Execute() path, so Armor, mana, suspension and budget gates apply
// identically.
type Invoker interface {
	Execute(ctx context.Context, req executor.ExecuteRequest) (executor.ExecuteResponse, error)
}

// Embedder is the subset of domain.Service the HTTP handlers call for
// embeddings. *domain.Service satisfies it directly (no mana decorator:
// embeddings self-ledger, unpriced).
type Embedder interface {
	Embed(ctx context.Context, req domain.EmbedFlowRequest) (domain.EmbedFlowResponse, error)
}

// Pinger is the minimal database-ping port for /readyz. *pg.Repo satisfies it.
type Pinger interface {
	Ping(ctx context.Context) error
}

// Settings carries the identity fallbacks the HTTP layer applies when the
// caller does not stamp them. Mirrors the Python gateway's Settings
// (default_tenant_id / default_gcid / default_agent_id).
type Settings struct {
	DefaultTenantID string
	DefaultGCID     string
	DefaultAgentID  string
}

// Server is the OpenAI-compatible HTTP server. It holds the Executor
// (Invoker), the raw domain service for embeddings (Embedder), the
// model registry, and the identity fallbacks.
type Server struct {
	invoker  Invoker
	embedder Embedder
	registry registry.Registry
	settings Settings
	db       Pinger
	now      func() time.Time
}

// NewServer constructs the OpenAI HTTP server. All dependencies are
// required (fail-loud per feedback_no_stubs_real_wiring): a nil Invoker or
// Embedder would silently bypass governance.
func NewServer(invoker Invoker, embedder Embedder, reg registry.Registry, db Pinger, settings Settings) (*Server, error) {
	if invoker == nil {
		return nil, errors.New("openai: Invoker required")
	}
	if embedder == nil {
		return nil, errors.New("openai: Embedder required")
	}
	if reg == nil {
		return nil, errors.New("openai: Registry required")
	}
	now := time.Now
	return &Server{
		invoker:  invoker,
		embedder: embedder,
		registry: reg,
		settings: settings,
		db:       db,
		now:      now,
	}, nil
}

// Handler returns the http.Handler with all routes registered. The caller
// mounts it on the HTTP server (cmd/server/main.go).
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()

	// Probes
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	mux.HandleFunc("GET /readyz", s.handleReadyz)

	// Model catalogue
	mux.HandleFunc("GET /v1/models", s.handleListModels)
	mux.HandleFunc("GET /v1/models/{model}", s.handleGetModel)

	// Chat completions (OpenAI + legacy path)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChat)
	mux.HandleFunc("POST /v1/completions", s.handleChat)

	// Responses (hosted web search surface)
	mux.HandleFunc("POST /v1/responses", s.handleResponses)

	// Image generation
	mux.HandleFunc("POST /v1/images/generations", s.handleImages)
	mux.HandleFunc("POST /v1/images/edits", s.handleImages)

	// Embeddings
	mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)

	// Method guards — catch-all for wrong methods on every route. Go 1.22+
	// ServeMux prefers the more specific method+path pattern, so a wrong
	// method falls through to these and gets the OpenAI 405 envelope.
	mux.HandleFunc("/v1/chat/completions", s.methodNotAllowed("POST"))
	mux.HandleFunc("/v1/completions", s.methodNotAllowed("POST"))
	mux.HandleFunc("/v1/responses", s.methodNotAllowed("POST"))
	mux.HandleFunc("/v1/images/generations", s.methodNotAllowed("POST"))
	mux.HandleFunc("/v1/images/edits", s.methodNotAllowed("POST"))
	mux.HandleFunc("/v1/embeddings", s.methodNotAllowed("POST"))
	mux.HandleFunc("/v1/models", s.methodNotAllowed("GET"))
	mux.HandleFunc("/v1/models/{model}", s.methodNotAllowed("GET"))
	mux.HandleFunc("/healthz", s.methodNotAllowed("GET"))
	mux.HandleFunc("/readyz", s.methodNotAllowed("GET"))

	return mux
}

// ---------------------------------------------------------------------------
// Response writers — every response carries the nosniff header
// ---------------------------------------------------------------------------

func writeJSON(w http.ResponseWriter, status int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(data)
}

func writeText(w http.ResponseWriter, status int, text string) {
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.WriteHeader(status)
	_, _ = fmt.Fprint(w, text)
}

// openaiError is the OpenAI error envelope shape.
type openaiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}

// errorResponse is the top-level error body: {"error": {...}}.
type errorResponse struct {
	Error openaiError `json:"error"`
}

func writeError(w http.ResponseWriter, status int, errType, code, message string) {
	writeJSON(w, status, errorResponse{
		Error: openaiError{
			Message: message,
			Type:    errType,
			Code:    code,
		},
	})
}

// ---------------------------------------------------------------------------
// Method guard
// ---------------------------------------------------------------------------

// methodNotAllowed returns a handler that answers 405 with the OpenAI error
// envelope. The allowed-method string is echoed in the message ("POST only").
func (s *Server) methodNotAllowed(allowed string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method_not_allowed", allowed+" only")
	}
}

// ---------------------------------------------------------------------------
// Probes
// ---------------------------------------------------------------------------

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	writeText(w, http.StatusOK, "ok")
}

func (s *Server) handleReadyz(w http.ResponseWriter, r *http.Request) {
	if s.db != nil {
		if err := s.db.Ping(r.Context()); err != nil {
			writeText(w, http.StatusServiceUnavailable, "not ready")
			return
		}
	}
	writeText(w, http.StatusOK, "ready")
}

// ---------------------------------------------------------------------------
// Model catalogue
// ---------------------------------------------------------------------------

func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	specs := s.registry.List()
	data := make([]map[string]any, 0, len(specs))
	for _, spec := range specs {
		data = append(data, modelEntry(spec))
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	model := r.PathValue("model")
	spec, ok := s.registry.Get(model)
	if !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error", "model_not_found",
			fmt.Sprintf("model \"%s\" is not in the registry", model))
		return
	}
	writeJSON(w, http.StatusOK, modelEntry(spec))
}

// modelEntry renders a registry spec as an OpenAI model object, mirroring
// the Python compat.model_entry: id / object / created / owned_by plus the
// additive metadata (context_window, max_output_tokens, capabilities).
//
// INFORMATION DISCLOSURE: the model list only exposes what the caller is
// allowed to invoke. It does NOT leak:
//   - base_url — provider configuration (the vendor endpoint is internal)
//   - credential availability — whether a credential is configured
//   - internal fallback names — the FallbackIDs chain is internal
//   - registry metadata — pricing, aliases, upstream model names
func modelEntry(spec registry.ModelSpec) map[string]any {
	out := map[string]any{
		"id":       spec.ID,
		"object":   "model",
		"created":  0,
		"owned_by": spec.Provider,
	}
	if spec.ContextWindow > 0 {
		out["context_window"] = spec.ContextWindow
	}
	if spec.MaxOutputTokens > 0 {
		out["max_output_tokens"] = spec.MaxOutputTokens
	}
	if len(spec.Capabilities) > 0 {
		out["capabilities"] = spec.Capabilities
	}
	return out
}
