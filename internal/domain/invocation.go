package domain

// InvokeRequest is the wire-level request type the gRPC and HTTP adapters have
// always built. It is an alias for the canonical ExecuteRequest — the
// Executor's single entry-point type — so existing adapter code keeps working
// while the governance layer speaks only ExecuteRequest.
type InvokeRequest = ExecuteRequest

// InvokeResponse is the wire-level response type the gRPC and HTTP adapters
// have always consumed. It is an alias for the canonical ExecuteResponse — the
// Executor's single entry-point type — so existing adapter code keeps working
// while the governance layer speaks only ExecuteResponse.
type InvokeResponse = ExecuteResponse

// TokenUsage is the canonical token / cost accounting struct. Every vendor
// adapter populates the fields its provider reports; the rest stay zero.
// This is the single canonical usage type — there is no separate Usage struct.
type TokenUsage struct {
	// InputTokens is the total prompt-side token count (including cached).
	InputTokens int64
	// CachedTokens is the subset of InputTokens served from cache.
	CachedTokens int64
	// CacheWriteTokens is the number of tokens written to cache this call.
	CacheWriteTokens int64
	// OutputTokens is the completion-side token count.
	OutputTokens int64
	// ReasoningTokens is the subset of OutputTokens spent on chain-of-thought.
	ReasoningTokens int64
	// ImageUnits is the number of generated images (image-modality calls).
	ImageUnits int64
	// EmbeddingUnits is the number of embedding vectors produced.
	EmbeddingUnits int64
	// GroundingUnits is the number of grounded web-search calls made.
	GroundingUnits int64
	// CostMicros is the provider's own cost in micro-USD (USD * 1e-6).
	CostMicros int64
}

// FinishReason mirrors chora.services.model_gateway.v1.FinishReason.
type FinishReason int

const (
	FinishReasonUnspecified     FinishReason = 0
	FinishReasonComplete        FinishReason = 1
	FinishReasonMaxTokens       FinishReason = 2
	FinishReasonModelArmorBlock FinishReason = 3
	FinishReasonBudgetBlock     FinishReason = 4
	FinishReasonVendorError     FinishReason = 5
	// FinishReasonManaBlock — per-user mana exhausted (WS-1 umbrella metering,
	// ADR-142 §4). Flows in-band like budget/armor blocks; FinishDetail carries
	// the structured upsell the FE 402 modal renders.
	FinishReasonManaBlock FinishReason = 6
)

// VendorRequest is the contract between the Executor layer and the provider
// ports (Provider.Generate). The Executor builds this from the
// resolved policy + the original ExecuteRequest.
type VendorRequest struct {
	Vendor         VendorFamily
	LogicalModelID LogicalModelID
	Prompt         string
	// ResponseModality is the requested output modality ("" / "TEXT" /
	// "IMAGE"). Vendor adapters that support image generation translate
	// "IMAGE" into the vendor-native modality selector (W8, CR 2026-06-01).
	ResponseModality string
	SystemPrompt     string
	GenerationConfig map[string]any
	Traceparent      string
	Tracestate       string
	TenantID         string // for BYOA secret resolution downstream
	// ContentsJSON / ToolsJSON carry the ADR-177 tool-calling conversation +
	// declarations through to the vendor adapter. When ContentsJSON is non-empty
	// the adapter builds the request from it (ignoring Prompt).
	ContentsJSON string
	ToolsJSON    string
}

// VendorResponse is what the Provider port returns to the Executor.
type VendorResponse struct {
	Completion string
	// ImageBytes / ImageMIMEType carry an inline image returned by the
	// vendor (e.g. base64-decoded PNG from a gemini image model). Empty
	// for text-only responses (W8, CR 2026-06-01).
	ImageBytes    []byte
	ImageMIMEType string
	Usage         TokenUsage
	ModelVersion  string // resolved by the vendor (post-optimizer)
	FinishReason  FinishReason
	FinishDetail  string
	// ToolCallsJSON carries model-emitted function calls (ADR-177); empty for a
	// terminal text turn.
	ToolCallsJSON string
	// Citations are the sources the model cited in its completion (Anthropic
	// text-block citations, OpenAI url_citation annotations). Empty when the
	// vendor returned none.
	Citations []Citation
	// SearchQueries are the web-search queries the model issued (Anthropic
	// server_tool_use blocks, OpenAI web_search_call output items). Empty on
	// ungrounded turns.
	SearchQueries []string
	// RevisedPrompt carries the provider-revised prompt for image generation
	// (OpenAI images/generations revised_prompt field). Empty for text turns.
	RevisedPrompt string
}

// Citation is one source a grounded/citing model completion referenced.
// Mirrors the Python runtime's citation dict (url/title/start_index/end_index).
type Citation struct {
	// URL is the cited source location. REQUIRED — a citation with no URL is
	// dropped by the adapter.
	URL string
	// Title is the source page title (may be empty).
	Title string
	// Snippet is the cited text segment (best-effort; may be empty).
	Snippet string
	// StartIndex / EndIndex are the character offsets of the cited span in
	// the completion text (OpenAI url_citation annotations). Zero when the
	// vendor did not report them.
	StartIndex int
	EndIndex   int
}
