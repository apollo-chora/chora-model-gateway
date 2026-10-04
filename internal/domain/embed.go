// embed.go — the Embed flow's domain types.
//
// Embeddings are the chokepoint's text-to-vector surface. Posture: no price
// is attached (a priced embedding would be user-visible) and no generation
// guardrail runs (text-to-vector has no completion surface), but the ledger
// row and the tenant/agent attribution are the point of routing it here.
package domain

import "time"

// DefaultEmbeddingModelID is the resolved model when the caller leaves
// logical_model_id empty. Any registry entry advertising the "embeddings"
// capability may be named instead.
const DefaultEmbeddingModelID LogicalModelID = "text-embedding-004"

// defaultEmbeddingDimensions is the fallback vector width when the caller
// does not ask for a specific size.
const defaultEmbeddingDimensions = 768

// EmbedRequest is the domain-side input for one text embedding.
type EmbedRequest struct {
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

// EmbedResponse is the domain-side output.
type EmbedResponse struct {
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
	// Target is the resolved upstream destination, including the credential.
	Target           TargetModel
	Credential       string
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
