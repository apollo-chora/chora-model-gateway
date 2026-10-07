// provider.go — the canonical vendor-provider contract (Python port).
//
// The Python reference (app/runtime.py) dispatches through a single Runtime
// with generate / image / embed methods returning one Result shape. This file
// ports that contract to Go: a Provider interface with capability-specific
// methods, one capability interface per method, and canonical request /
// response / usage types that every vendor adapter translates to and from.
//
// The governance layer (Armor, mana, suspension, budget) stays in the domain
// service — providers are dumb adapters beneath it. A vendor adapter
// implements the capability interfaces it actually supports; the full
// Provider interface is the union of all four capabilities.
package domain

import (
	"context"
	"fmt"
)

// ProviderError is the canonical error type for a non-2xx upstream response.
// Carries the provider's own HTTP status so the service + adapter can relay
// it (401/403 → Unauthenticated, 404 → NotFound, 429 → ResourceExhausted,
// else Unavailable) instead of flattening every failure into one code.
//
// This is the Go port of the Python runtime.ProviderError. Each vendor
// adapter's local ProviderError is replaced by this canonical type.
type ProviderError struct {
	// Status is the provider's own HTTP status code.
	Status int
	// Endpoint is the full URL that produced the error.
	Endpoint string
	// Body is the (truncated) response body for diagnostics.
	Body string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("upstream %d from %s: %s", e.Status, e.Endpoint, e.Body)
}

// UpstreamStatus exposes the provider's HTTP status for status mapping.
// Implements the UpstreamStatusProvider port consumed by Service.Invoke.
func (e *ProviderError) UpstreamStatus() int { return e.Status }

// GroundingConfig carries grounding-specific request parameters. Nil (or
// Enabled=false) means an ungrounded call.
type GroundingConfig struct {
	// Enabled requests "Grounding with Google Search" for this call.
	Enabled bool
	// MaxResults caps the number of citations the model may return.
	MaxResults int32
}

// GenerateRequest is the canonical request for text generation. Vendor
// adapters translate this to their native wire format.
type GenerateRequest struct {
	// Model is the concrete vendor model id (e.g. "gemini-2.5-pro").
	Model string
	// Prompt is the user-facing prompt text.
	Prompt string
	// SystemPrompt is the system instruction (platform-controlled, not screened).
	SystemPrompt string
	// ContentsJSON is the multi-turn conversation as JSON (genai []*Content).
	// When non-empty the adapter builds the request from it, ignoring Prompt.
	ContentsJSON string
	// ToolsJSON is the tool/function declarations as JSON (genai []*Tool).
	ToolsJSON string
	// GenerationConfig carries vendor-neutral generation parameters
	// (temperature, top_p, max_tokens, ...).
	GenerationConfig map[string]any
	// Grounding carries grounding-specific parameters. Nil = ungrounded.
	Grounding *GroundingConfig
}

// GenerateResponse is the canonical response from text generation.
type GenerateResponse struct {
	// Text is the completion text.
	Text string
	// Citations are the sources the model cited in its completion.
	Citations []Citation
	// SearchQueries are the web-search queries the model issued.
	SearchQueries []string
	// FinishReason is the canonical finish reason.
	FinishReason FinishReason
	// Usage is the token / cost accounting for this call.
	Usage TokenUsage
	// Vendor is the vendor family identifier (e.g. "vertex_ai_gemini").
	Vendor string
	// ModelVersion is the concrete model version the vendor resolved.
	ModelVersion string
	// RevisedPrompt is the provider-revised prompt (image generation).
	RevisedPrompt string
	// ToolCallsJSON carries model-emitted function calls (ADR-177).
	ToolCallsJSON string
}

// EmbedRequest is the canonical request for text embeddings.
type EmbedRequest struct {
	// Model is the concrete embedding model id.
	Model string
	// Text is the text to embed.
	Text string
	// OutputDimensions is the desired vector dimension (0 = model default).
	OutputDimensions int32
}

// EmbedResponse is the canonical response from text embeddings.
type EmbedResponse struct {
	// Values is the embedding vector.
	Values []float32
	// ModelVersion is the concrete model version used.
	ModelVersion string
	// Usage is the token / cost accounting for this call.
	Usage TokenUsage
}

// ImageRequest is the canonical request for image generation.
type ImageRequest struct {
	// Model is the concrete image model id.
	Model string
	// Prompt is the image-generation prompt.
	Prompt string
	// GenerationConfig carries vendor-neutral generation parameters.
	GenerationConfig map[string]any
}

// ImageResponse is the canonical response from image generation.
type ImageResponse struct {
	// ImageBytes is the generated image (e.g. PNG).
	ImageBytes []byte
	// ImageMIMEType is the MIME type of ImageBytes (e.g. "image/png").
	ImageMIMEType string
	// RevisedPrompt is the provider-revised prompt (safety-modified).
	RevisedPrompt string
	// Usage is the token / cost accounting for this call.
	Usage TokenUsage
	// ModelVersion is the concrete model version used.
	ModelVersion string
}

// GroundedGenerateRequest is the canonical request for grounded search.
type GroundedGenerateRequest struct {
	// Model is the concrete grounding-capable model id.
	Model string
	// Prompt is the screened learner directive.
	Prompt string
	// SystemPrompt optionally carries a citation-mandate system instruction.
	SystemPrompt string
	// MaxResults caps the number of citations.
	MaxResults int32
}

// GroundedGenerateResponse is the canonical response from grounded search.
type GroundedGenerateResponse struct {
	// Text is the grounded completion.
	Text string
	// Citations are the structured sources parsed from groundingMetadata.
	Citations []GroundedCitation
	// SearchQueries are the queries the model issued.
	SearchQueries []string
	// Usage is the token / cost accounting for this call.
	Usage TokenUsage
	// ModelVersion is the concrete model version used.
	ModelVersion string
	// FinishReason is the canonical finish reason.
	FinishReason FinishReason
}

// TextGenerator is the capability interface for text generation.
type TextGenerator interface {
	Generate(ctx context.Context, req GenerateRequest) (*GenerateResponse, error)
}

// Embedder is the capability interface for text embeddings.
type Embedder interface {
	Embed(ctx context.Context, req EmbedRequest) (*EmbedResponse, error)
}

// ImageGenerator is the capability interface for image generation.
type ImageGenerator interface {
	GenerateImage(ctx context.Context, req ImageRequest) (*ImageResponse, error)
}

// GroundedGenerator is the capability interface for grounded search.
type GroundedGenerator interface {
	GroundedGenerate(ctx context.Context, req GroundedGenerateRequest) (*GroundedGenerateResponse, error)
}

// CapabilityProvider is the full vendor capability interface — the union of
// all four capability interfaces. A vendor adapter implements the capabilities
// it actually supports; the service layer type-asserts to the specific
// capability interface it needs.
//
// This is distinct from the Provider dispatch interface (executor.go) the
// Executor drives: CapabilityProvider is the canonical four-surface contract
// (text / embed / image / grounded), while Provider is the single
// Family + Generate dispatch method the governance layer calls.
type CapabilityProvider interface {
	TextGenerator
	Embedder
	ImageGenerator
	GroundedGenerator
}
