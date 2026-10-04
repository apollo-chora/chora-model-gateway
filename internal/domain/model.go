// Package domain contains the pure domain types + business logic for
// chora-model-gateway. ZERO infrastructure imports — no net/http, no
// google.golang.org/grpc, no database/sql, no anything from the adapter
// side. Compose against the ports interfaces only.
//
// Hexagonal direction-of-dependency: adapter → port → domain. NEVER reverse.
package domain

// VendorFamily identifies the wire protocol a dispatch uses. It selects the
// adapter implementation; it is NOT the same thing as the model name, which
// the caller addresses as a LogicalModelID and the registry resolves into a
// TargetModel.
//
// Two families are supported, chosen because the overwhelming majority of
// self-hosted and hosted inference servers speak the first:
//   - VendorFamilyOpenAI: the OpenAI chat-completions / images / embeddings
//     shape. Covers OpenAI itself plus vLLM, Ollama, LM Studio, llama.cpp's
//     server, Groq, Together, Fireworks, and any other adopter of the spec.
//   - VendorFamilyAnthropic: the Anthropic messages shape.
type VendorFamily string

const (
	VendorFamilyOpenAI    VendorFamily = "openai"
	VendorFamilyAnthropic VendorFamily = "anthropic"
)

// LogicalModelID is the caller-supplied identifier the gateway resolves to a
// vendor + concrete upstream target via the model registry. It is whatever
// the registry names — "gpt-4o", "llama3.1:8b", "chora-fast" are all valid.
type LogicalModelID string
