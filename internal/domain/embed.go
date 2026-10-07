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
	LogicalModelID   LogicalModelID
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
