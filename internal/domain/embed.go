// embed.go — the Embed flow's domain types (G1'-1, owner-ruled 2026-08-07).
//
// Embeddings were the chokepoint's one true bypass: ADR-173 sanctioned
// direct Vertex `:predict` calls from chora-consumption and chora-creation
// because this service had no embeddings RPC. This closes it. Posture per
// the closing rule: ENTER AT PERMISSIVE, tighten nothing. No Armor leg runs
// (text-to-vector has no generation surface); the ledger row and the
// tenant/agent attribution are the point of the re-route; no price is
// attached (a priced embedding would be user-visible).
package domain

import "time"

// DefaultEmbeddingModelID is the resolved model when the caller leaves
// logical_model_id empty. Matches both former direct call sites.
const DefaultEmbeddingModelID LogicalModelID = "text-embedding-004"

// defaultEmbeddingDimensions matches every pgvector(768) column both call
// sites write (atom_embeddings, familiar_memory_recall).
const defaultEmbeddingDimensions = 768

// embeddingModelAllowlist enumerates the model ids the Embed flow accepts.
// A generation model on the embed path is refused, never silently
// re-routed; extending this list is a reviewed edit.
var embeddingModelAllowlist = map[LogicalModelID]bool{
	DefaultEmbeddingModelID:           true,
	"text-embedding-005":              true,
	"text-multilingual-embedding-002": true,
}

// EmbeddingRoutePin pins the Embed flow to ONE logical model id for a
// demo-specific deployment. When wired, a caller-supplied model that is not
// the pinned id is refused rather than dispatched, so a runtime override
// cannot redirect the demo's embedding route to another (potentially paid)
// model. nil means unpinned — the historical behaviour.
//
// The pin is deliberately an exact-id assertion, not a name-shape rule: the
// same guard that proves the demo route also proves nothing about billing for
// any other model, and a name is not an authoritative billing policy. The
// matching upstream_model assertion lives in the startup registry check
// (cmd/server/embedding_pin.go).
type EmbeddingRoutePin struct {
	LogicalID LogicalModelID
}

// EmbedFlowRequest is the domain-side input for one text embedding.
type EmbedFlowRequest struct {
	InvocationID     string
	TenantID         string
	GCID             string
	AgentID          string
	CrewKind         string
	LogicalModelID   LogicalModelID
	Text             string
	TaskType         string
	OutputDimensions int32
	Traceparent      string
	Tracestate       string
}

// EmbedFlowResponse is the domain-side output.
type EmbedFlowResponse struct {
	InvocationID   string
	Values         []float32
	Vendor         string
	ModelVersion   string
	Usage          TokenUsage
	LatencyMs      int32
	CompletedAt    time.Time
	GatewayVersion string
}

// EmbedVendorRequest is what the embedding vendor adapter receives.
type EmbedVendorRequest struct {
	// LogicalModelID is the caller's logical model id (an alias).
	LogicalModelID LogicalModelID

	// UpstreamModel is the resolved registry upstream_model — the model name
	// the provider actually speaks. Empty means LogicalModelID, which is the
	// behaviour of a dispatcher that has no registry to resolve against.
	UpstreamModel string

	// BaseURL is the resolved registry base_url — the API root to dispatch
	// to. Empty means the adapter's own configured endpoint.
	BaseURL string

	Text             string
	TaskType         string
	OutputDimensions int32
	TenantID         string
	Traceparent      string
}

// EmbedVendorResponse is what the embedding vendor adapter returns.
// InputTokens carries the vendor-reported statistics token_count (0 when
// the vendor omits it).
type EmbedVendorResponse struct {
	Values       []float32
	ModelVersion string
	InputTokens  int64
}
