// Package domain contains the pure domain types + business logic for
// chora-model-gateway per ADR-163. ZERO infrastructure imports — no
// google.golang.org/grpc, no cloud.google.com/*, no jackc/pgx, no anything
// from the adapter side. Compose against the ports interfaces only.
//
// Hexagonal direction-of-dependency: adapter → port → domain. NEVER reverse.
package domain

// VendorFamily identifies the high-level vendor category for routing +
// adapter selection. Maps to one of the four adapter packages under
// internal/adapter/vendor/.
type VendorFamily string

const (
	VendorFamilyVertexGemini VendorFamily = "vertex_ai_gemini"
	VendorFamilyVertexGemma  VendorFamily = "vertex_ai_gemma"
	VendorFamilyOpenAI       VendorFamily = "openai_byoa"
	VendorFamilyAnthropic    VendorFamily = "anthropic_byoa"
)

// LogicalModelID is the caller-supplied identifier the gateway resolves
// to a vendor + concrete model version via the per-agent routing policy.
// Examples: "gemini-2.5-pro", "gemma-sg-academic", "openai/gpt-4o".
type LogicalModelID string
