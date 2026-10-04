package domain

import "time"

// InvokeRequest is the domain-shaped input to gateway.Service.Invoke.
// Lives in the domain so the service layer carries no proto dependency.
// The grpc adapter maps proto → domain (and back) at the wire boundary;
// the OpenAI-compatible HTTP adapter does the same for the /v1 surface.
type InvokeRequest struct {
	// Caller-supplied UUIDv7. If empty, the service generates one + echoes
	// in the response. Idempotency key for ledger dedup.
	InvocationID string

	// REQUIRED — tenant scope. RLS-enforced inside BudgetRepo
	// (SET LOCAL chora.tenant_id = ...). UUIDv7.
	TenantID string

	// REQUIRED — actor identity (learner GCID or agent AGID). UUIDv7.
	GCID string

	// REQUIRED — calling agent identifier (a crew name, a graph-node id, or
	// "openai_compat" for the HTTP facade).
	AgentID string

	// OPTIONAL — crew classification used as fallback policy key.
	CrewKind string

	// REQUIRED — logical model identifier (e.g. "gpt-4o", "llama3.1:8b").
	// Must name an entry in the model registry; an unknown id fails loud.
	LogicalModelID LogicalModelID

	// OPTIONAL — agent-declared ordered fallback chain of logical model ids,
	// tried in order on primary dispatch failure. Each id is resolved
	// through the registry exactly like the primary, so a chain may cross
	// providers. Empty = single-shot.
	FallbackModelIDs []LogicalModelID

	// REQUIRED — user-facing prompt content.
	Prompt string

	// OPTIONAL — output modality selector: "" (default) / "TEXT" / "IMAGE" /
	// "GROUNDED".
	//
	// "IMAGE" routes to an entry advertising the "image" capability and
	// returns bytes. "GROUNDED" routes to an entry advertising the
	// "web_search" capability and runs the provider's hosted web search
	// during the call, returning the answer with normalised citations.
	//
	// Both are refused against an entry that does not advertise the matching
	// capability. The budget, ledger and fallback machinery is
	// response-agnostic, so a modality only changes the vendor request shape
	// and the response carrier — not the orchestration.
	ResponseModality string

	// OPTIONAL — system-instruction prompt.
	SystemPrompt string

	// OPTIONAL — free-text role tag used for cost attribution on the ledger
	// row and in the logs (e.g. "atom_validator", "router"). On the
	// OpenAI-compatible facade this defaults to the caller's `user` field or
	// "openai_compat".
	ActionCode string

	// OPTIONAL — tool-calling conversation: JSON of a multi-turn message
	// history (OpenAI `messages` shape, or the genai []*Content shape on
	// the gRPC surface). When non-empty the vendor adapter builds the model
	// request from THIS and ignores Prompt. Empty = text-only path.
	ContentsJSON string

	// OPTIONAL — tool/function declarations (JSON). Empty = plain generation.
	ToolsJSON string

	// OPTIONAL — vendor-neutral generation parameters (temperature, top_p,
	// max_tokens, stop, ...). The gateway applies the registry's
	// MaxOutputTokens ceiling over whatever the caller sent.
	GenerationConfig map[string]any

	// OPTIONAL — the surface the call is attributed to. Recorded on the
	// ledger row; empty is allowed and recorded as "unspecified".
	Surface string

	// OPTIONAL — an idempotency key for the dispatch. When set, the gateway
	// takes a keyed claim so a redelivered dispatch bills once. Empty on
	// non-dispatched calls.
	DispatchIdempotencyKey string

	// OPTIONAL — W3C traceparent.
	Traceparent string

	// OPTIONAL — W3C tracestate.
	Tracestate string

	// ExtraHeaders are provider-specific headers to forward (e.g. an
	// OpenRouter referer, a HuggingFace routing hint). Sourced from the
	// registry entry, never from the caller's HTTP headers.
	ExtraHeaders map[string]string
}

// InvokeResponse is the domain-shaped output of gateway.Service.Invoke.
type InvokeResponse struct {
	InvocationID string
	Completion   string

	// ImageBytes carries the generated image (e.g. PNG) for image-modality
	// responses; empty for text. ImageMIMEType is the corresponding MIME
	// type (e.g. "image/png").
	ImageBytes    []byte
	ImageMIMEType string

	// ImageRevisedPrompt carries a provider's rephrasing of the image
	// prompt, when the upstream returns one (OpenAI's `revised_prompt`).
	ImageRevisedPrompt string

	Usage        TokenUsage
	Vendor       string // resolved provider family (post-fallback)
	ModelVersion string // the upstream model name actually dispatched

	FallbackChain  []string // one entry per attempted vendor:model
	LatencyMs      int32
	FinishReason   FinishReason
	FinishDetail   string
	CompletedAt    time.Time
	GatewayVersion string

	// ToolCallsJSON carries the model-emitted function calls this turn.
	// Non-empty ⇒ the agent must execute the tools + continue the loop;
	// empty ⇒ terminal text turn.
	ToolCallsJSON string

	// Citations are the sources a GROUNDED call consulted, normalised across
	// providers, flattened across every message. Nil for text and image
	// responses. On a grounded response an EMPTY slice is meaningful: the model
	// searched and produced nothing citable. Prefer Messages when you need to
	// know WHICH claim a source supports.
	Citations []GroundingCitation

	// Messages is the ordered assistant messages of a multi-message reply.
	// A reasoning model narrates between its searches and answers last, so this
	// is usually more than one entry. Nil when the provider sent a single
	// message, which is the case for every non-reasoning model and for chat
	// completions.
	Messages []GroundingMessage

	// SearchQueries are the queries the provider actually ran.
	SearchQueries []string

	// GroundingSurface names which search endpoint served this call
	// ("responses", "chat_completions", "messages"). Empty when the call was
	// not grounded. Reported so an operator debugging a grounding problem knows
	// which of the three shapes actually went on the wire.
	GroundingSurface string
}

// TokenUsage is the wire-equivalent of the proto TokenUsage sub-message.
type TokenUsage struct {
	InputTokens  int64
	OutputTokens int64
	CachedTokens int64
	CostMicros   int64 // USD * 1e-6
}

// FinishReason mirrors chora.services.model_gateway.v1.FinishReason. The
// numeric values are pinned to the shared proto enum so the gRPC surface
// keeps translating without a lookup table.
type FinishReason int

const (
	FinishReasonUnspecified FinishReason = 0
	FinishReasonComplete    FinishReason = 1
	FinishReasonMaxTokens   FinishReason = 2
	// FinishReasonContentBlock is the retired Cloud Model Armor value. It is
	// retained so a v1 client decoding an old numeric value does not silently
	// reinterpret it; the gateway never emits it.
	FinishReasonContentBlock FinishReason = 3
	FinishReasonBudgetBlock  FinishReason = 4
	FinishReasonVendorError  FinishReason = 5
	// FinishReasonMeteringBlock is the retired identity-ManaService value,
	// retained for the same reason as FinishReasonContentBlock.
	FinishReasonMeteringBlock FinishReason = 6
)

// VendorRequest is the contract between the service layer and the vendor
// ports (VendorClient.Generate). The service builds this from the resolved
// policy + the original InvokeRequest.
type VendorRequest struct {
	// Target is the fully-resolved upstream destination, including the
	// credential. One adapter instance serves every registry entry that
	// shares a provider.
	Target TargetModel

	Prompt           string
	ResponseModality string
	SystemPrompt     string

	GenerationConfig map[string]any
	ExtraHeaders     map[string]string

	TenantID string

	Traceparent string
	Tracestate  string

	// ContentsJSON / ToolsJSON carry the tool-calling conversation +
	// declarations through to the vendor adapter. When ContentsJSON is
	// non-empty the adapter builds the request from it (ignoring Prompt).
	ContentsJSON string
	ToolsJSON    string

	// Credential is the resolved API key for this dispatch. It travels on
	// the request rather than on the adapter so two registry entries sharing
	// a provider can hold different keys. Empty means the target needs no
	// credential.
	Credential string
}

// VendorResponse is what the VendorClient port returns to the service.
type VendorResponse struct {
	Completion string

	// ImageBytes / ImageMIMEType carry an inline image returned by the
	// provider. Empty for text-only responses.
	ImageBytes         []byte
	ImageMIMEType      string
	ImageRevisedPrompt string

	Usage        TokenUsage
	ModelVersion string
	FinishReason FinishReason
	FinishDetail string

	// ToolCallsJSON carries model-emitted function calls; empty for a
	// terminal text turn.
	ToolCallsJSON string

	// Citations are the sources a GROUNDED call consulted, normalised across
	// providers. Empty for text and image responses — and, importantly, also
	// empty when a grounded call came back with NO citations, which is how a
	// caller tells "the model searched and found nothing citable" from "this
	// was never a grounded call".
	Citations []GroundingCitation

	// Messages is the ordered assistant messages of a multi-message reply. A
	// reasoning model narrates between its searches and answers LAST, so this
	// is usually more than one entry; Completion carries only that final answer.
	// Empty for a single-message reply, which is every non-reasoning model and
	// every chat-completions call.
	Messages []GroundingMessage

	// SearchQueries are the queries the provider actually ran. Useful when an
	// answer looks wrong: the query it searched is usually the culprit.
	SearchQueries []string
}
