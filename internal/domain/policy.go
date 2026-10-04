package domain

// AgentPolicy is the per-agent routing decision the policy loader resolves
// at every Invoke. The loader reads it from the model registry
// (internal/config) rather than from a Cloud Model Armor tier table: a
// registry entry names the provider, the upstream model, the endpoint and
// the credential reference.
//
// The PolicyLoader port returns this struct given an agent_id +
// logical_model_id from the InvokeRequest.
type AgentPolicy struct {
	// Logical agent identifier (a crew name like "qgen_question", a graph
	// node like "ai_assist_crew.broker_call", or — on the OpenAI-compatible
	// facade — the client-supplied agent, defaulting to "openai_compat").
	AgentID string

	// Resolved canonical logical model id. May differ from the caller's
	// request when the policy enforces a tier (e.g. every qgen_question
	// request resolves to the registry's designated model regardless of the
	// caller hint). Empty means "whatever the caller asked for", which is
	// what the OpenAI-compatible facade relies on.
	ResolvedLogicalModelID LogicalModelID

	// Target is the fully-resolved upstream destination: provider family,
	// upstream model name, base URL, endpoint paths and the credential
	// reference. Populated from the registry entry named by
	// ResolvedLogicalModelID.
	Target TargetModel

	// Ordered fallback chain to try on vendor 5xx or rate-limit. Empty =
	// single-shot.
	FallbackChain []AgentPolicyFallback
}

// AgentPolicyFallback is one entry in the routing fallback chain.
type AgentPolicyFallback struct {
	Vendor                 VendorFamily
	ResolvedLogicalModelID LogicalModelID
	Target                 TargetModel
}

// TargetModel is the fully-resolved upstream destination for one dispatch.
// It travels on VendorRequest so a single adapter instance serves every
// registry entry that shares a provider — the alternative (one adapter
// instance per endpoint) does not scale to a user-supplied registry.
type TargetModel struct {
	// Vendor selects the adapter implementation.
	Vendor VendorFamily

	// LogicalModelID is the registry id the caller addressed. Reported back
	// for ledger attribution.
	LogicalModelID LogicalModelID

	// UpstreamModel is the model name sent to the provider. Often equal to
	// LogicalModelID, but a registry entry may alias one name onto another
	// (e.g. expose "chora-fast" and dispatch "gpt-4o-mini").
	UpstreamModel string

	// BaseURL is the API root, e.g. "https://api.openai.com/v1" or
	// "http://host.docker.internal:11434/v1" for Ollama.
	BaseURL string

	// ChatCompletionsPath is appended to BaseURL for text/tool calls on an
	// OpenAI-shaped provider. Default "/chat/completions". An ABSOLUTE value
	// (one that starts with http:// or https://) is used verbatim instead of
	// being joined, which is how a fully custom chat-completions endpoint is
	// expressed. Ignored by the Anthropic provider, which uses MessagesPath.
	ChatCompletionsPath string

	// MessagesPath is the Anthropic-shaped equivalent of ChatCompletionsPath:
	// appended to BaseURL, default "/v1/messages". Absolute values are used
	// verbatim. Only the Anthropic adapter reads it.
	MessagesPath string

	// ImagesPath is appended to BaseURL for image generation.
	// Default "/images/generations".
	ImagesPath string

	// EmbeddingsPath is appended to BaseURL for embeddings.
	// Default "/embeddings".
	EmbeddingsPath string

	// APIKeyRef names where the credential comes from — an environment
	// variable name under the env-backed resolver. Empty means the target
	// needs no credential (a local vLLM / Ollama / LM Studio server).
	APIKeyRef string

	// ContextWindow is the model's total context in tokens (input + output).
	// Zero means unconstrained. Surfaced on GET /v1/models.
	ContextWindow int

	// MaxOutputTokens caps generation. Zero means unconstrained; the
	// gateway then forwards the caller's max_tokens untouched.
	MaxOutputTokens int

	// Capabilities gates which surfaces may address this model. Recognised
	// values: "chat", "tools", "vision", "image", "embeddings". An empty
	// list means chat only.
	Capabilities []string

	// ExtraHeaders are provider-specific headers to send on every dispatch to
	// this target (e.g. an OpenRouter referer, a HuggingFace routing hint).
	// Sourced from the registry file, never from caller-supplied HTTP headers.
	ExtraHeaders map[string]string

	// Grounding describes this entry's hosted web search. Nil means the entry
	// has none and a grounded request is refused — which is the right default:
	// dispatching a search tool to a provider that silently ignores it yields
	// an ungrounded answer that LOOKS grounded, and the caller has no way to
	// tell the difference.
	Grounding *Grounding
}

// Supports reports whether the target advertises the named capability.
func (t TargetModel) Supports(capability string) bool {
	if len(t.Capabilities) == 0 {
		return capability == "chat"
	}
	for _, c := range t.Capabilities {
		if c == capability {
			return true
		}
	}
	return false
}

// ChatCompletionsURL is the absolute URL for an OpenAI-shaped text/tool dispatch.
func (t TargetModel) ChatCompletionsURL() string {
	return resolveEndpoint(t.BaseURL, t.ChatCompletionsPath, "/chat/completions")
}

// MessagesURL is the absolute URL for an Anthropic-shaped text/tool dispatch.
func (t TargetModel) MessagesURL() string {
	return resolveEndpoint(t.BaseURL, t.MessagesPath, "/v1/messages")
}

// PrimaryDispatchURL is the endpoint this target's text surface actually
// uses, provider-aware. The boot log prints this, so an operator sees the URL
// that will really be called rather than an OpenAI-shaped guess at an
// Anthropic entry.
func (t TargetModel) PrimaryDispatchURL() string {
	if t.Vendor == VendorFamilyAnthropic {
		return t.MessagesURL()
	}
	return t.ChatCompletionsURL()
}

// ImagesURL is the absolute URL for an image-generation dispatch.
func (t TargetModel) ImagesURL() string {
	return resolveEndpoint(t.BaseURL, t.ImagesPath, "/images/generations")
}

// EmbeddingsURL is the absolute URL for an embedding dispatch.
func (t TargetModel) EmbeddingsURL() string {
	return resolveEndpoint(t.BaseURL, t.EmbeddingsPath, "/embeddings")
}

// GroundingURL is the absolute URL for a grounded dispatch, following the
// surface the entry configured. An entry with no Grounding block returns "",
// which the service turns into a refusal rather than a guess.
func (t TargetModel) GroundingURL() string {
	if t.Grounding == nil {
		return ""
	}
	surface := t.Grounding.EffectiveSurface(t.Vendor)
	switch surface {
	case SurfaceMessages:
		return t.MessagesURL()
	case SurfaceChatCompletions:
		return t.ChatCompletionsURL()
	default:
		return resolveEndpoint(t.BaseURL, t.Grounding.ResponsesPath, "/responses")
	}
}

// resolveEndpoint joins a base URL with a path, honouring an absolute path
// override. An empty base yields the path alone (a same-origin relative
// call), which is how a gateway fronted by a single reverse proxy is
// expressed.
func resolveEndpoint(base, path, defaultPath string) string {
	if path == "" {
		path = defaultPath
	}
	if isAbsoluteURL(path) {
		return path
	}
	if base == "" {
		return path
	}
	if path[0] == '/' {
		return trimTrailingSlash(base) + path
	}
	return trimTrailingSlash(base) + "/" + path
}

func isAbsoluteURL(s string) bool {
	return len(s) > 7 && (s[:7] == "http://" || (len(s) > 8 && s[:8] == "https://"))
}

func trimTrailingSlash(s string) string {
	for len(s) > 0 && s[len(s)-1] == '/' {
		s = s[:len(s)-1]
	}
	return s
}
