package domain

// Grounding describes how a registry entry's provider exposes hosted web
// search.
//
// It exists because there is no single "web search" wire format. OpenAI puts
// its tool on the Responses API (`{"type":"web_search"}`), Anthropic puts a
// differently-named tool on the Messages API
// (`{"type":"web_search_20250305","name":"web_search","max_uses":5}`), and the
// OpenAI-compatible aggregators pick either. The gateway normalises the
// CALLER's intent ("ground this answer in the web") into whatever the
// configured provider wants, so a client never has to know which it is
// talking to.
//
// A nil Grounding means the entry has no hosted search and the
// web_search capability is refused.
type Grounding struct {
	// Surface selects the endpoint the grounded call goes to.
	//
	//   responses        POST {responses_path}      (default for openai)
	//   chat_completions POST {chat_completions_path}
	//   messages         POST {messages_path}       (default for anthropic)
	Surface GroundingSurface

	// ResponsesPath is the OpenAI Responses endpoint. Default "/responses".
	ResponsesPath string

	// ToolType is the tool discriminator sent to the provider. Empty picks
	// the surface's default:
	//
	//	responses        -> "web_search"
	//	chat_completions -> "web_search_preview"
	//	messages         -> "web_search_20250305"
	//
	// Override it for a provider that renamed the tool — the aggregators
	// accept "web_search" on the Responses API, and a self-hosted gateway
	// may accept anything at all.
	ToolType string

	// ToolName is Anthropic's `name` field, which their server tools require.
	// Empty defaults to "web_search".
	ToolName string

	// MaxUses caps how many searches one call may make. Anthropic prices per
	// search and requires this field; OpenAI ignores it. Zero = provider
	// default.
	MaxUses int

	// ExtraToolFields are merged into the tool object verbatim, for provider
	// options this struct does not model (domain filters, user location,
	// search_context_size). Sourced from the registry, never from the caller.
	ExtraToolFields map[string]any
}

// GroundingSurface is the endpoint family a grounded call is dispatched to.
type GroundingSurface string

const (
	// SurfaceResponses is the OpenAI Responses API. It is the only surface
	// that supports the hosted web_search tool with full controls, and the
	// one the OpenAI-compatible aggregators implement.
	SurfaceResponses GroundingSurface = "responses"

	// SurfaceChatCompletions is the classic chat-completions surface with a
	// search tool attached. Fewer controls, wider support.
	SurfaceChatCompletions GroundingSurface = "chat_completions"

	// SurfaceMessages is the Anthropic Messages API with its server tool.
	SurfaceMessages GroundingSurface = "messages"
)

// EffectiveSurface resolves the surface, defaulting by provider family so a
// registry entry that says nothing still lands on the right endpoint.
func (g Grounding) EffectiveSurface(vendor VendorFamily) GroundingSurface {
	if g.Surface != "" {
		return g.Surface
	}
	if vendor == VendorFamilyAnthropic {
		return SurfaceMessages
	}
	return SurfaceResponses
}

// EffectiveToolType resolves the tool discriminator for the surface.
func (g Grounding) EffectiveToolType(surface GroundingSurface) string {
	if g.ToolType != "" {
		return g.ToolType
	}
	switch surface {
	case SurfaceMessages:
		return "web_search_20250305"
	case SurfaceChatCompletions:
		return "web_search_preview"
	default:
		return "web_search"
	}
}

// EffectiveToolName resolves Anthropic's required `name` field.
func (g Grounding) EffectiveToolName() string {
	if g.ToolName != "" {
		return g.ToolName
	}
	return "web_search"
}

// GroundingMessage is one assistant message from a multi-turn provider reply.
//
// The Responses API returns SEVERAL `message` items for a single grounded call:
// a reasoning model narrates between its searches ("I'll look that up", "the
// results conflict, so I'll open the other page") and only then states the
// answer. Collapsing those into one string produces narration soup with no
// answer in it, so they are carried as an ordered list and the answer is the
// LAST one.
//
// The `output_text` parts of a single message item are still joined together —
// within one item they are genuinely one answer.
type GroundingMessage struct {
	// Text is this message's content, with its output_text parts joined.
	Text string

	// Citations are the sources this specific message cites. Attaching them per
	// message rather than to the whole response is what lets a caller render
	// "claim X rests on source Y".
	Citations []GroundingCitation
}

// GroundingCitation is one source the provider consulted, normalised across
// providers.
//
// The providers disagree on almost everything here — OpenAI nests citations
// as `url_citation` annotations with character offsets on the output text,
// Anthropic returns a flat `citations` array on each text block with the cited
// passage attached — so a caller that wanted citations would otherwise need a
// branch per vendor. This is the shape both are parsed into.
type GroundingCitation struct {
	// URL is the source. The provider mints a redirect that expires (OpenAI's
	// about ~30 days out), so treat it as display-only and not a durable link.
	URL string `json:"url"`

	// Title is the page title, when the provider supplies one.
	Title string `json:"title,omitempty"`

	// Snippet is the cited passage. Anthropic provides this; OpenAI does not.
	Snippet string `json:"snippet,omitempty"`

	// StartIndex and EndIndex bound the citation within that message's text.
	// OpenAI supplies them; Anthropic leaves them zero, which is omitted from
	// the JSON rather than serialised as a misleading 0..0 range.
	StartIndex int `json:"start_index,omitempty"`
	EndIndex   int `json:"end_index,omitempty"`
}
