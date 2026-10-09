package domain

import "context"

// ModelInfo is the domain's view of a model's dispatch metadata — everything
// the Invoke flow needs from the model registry to dispatch a call against
// one model: what it can do, its output ceiling, its credentials, and its
// grounding configuration.
//
// The production adapter wraps internal/registry.ModelSpec; the domain
// carries no registry dependency (hexagonal: adapter → domain).
type ModelInfo struct {
	// ID is the canonical logical model identifier.
	ID string

	// Vendor is the vendor family this model dispatches to.
	Vendor VendorFamily

	// UpstreamModel is the model name sent to the provider. Empty means ID.
	UpstreamModel string

	// BaseURL is the resolved API root for this entry. Empty means the entry
	// declares none and no role endpoint is configured for it — a route that
	// cannot be dispatched to a verified endpoint.
	BaseURL string

	// Capabilities advertises what the model can do ("chat", "tools",
	// "vision", "image", "embeddings", "web_search"). Empty means ["chat"].
	Capabilities []string

	// MaxOutputTokens is the generation ceiling; the gateway clamps callers
	// to it. 0 = unbounded.
	MaxOutputTokens int

	// APIKeyEnv is the env var holding the credential. Empty = no auth.
	APIKeyEnv string

	// APIKey is the resolved credential (from the environment). Empty when
	// APIKeyEnv is unset or resolves to an empty value.
	APIKey string

	// Grounding is the hosted web search configuration. Nil = the model
	// refuses grounded requests.
	Grounding *GroundingInfo

	// FallbackIDs are the logical model IDs tried in order when this model
	// fails. Empty = no fallback.
	FallbackIDs []LogicalModelID
}

// GroundingInfo is the domain's view of a model's hosted web search
// configuration.
type GroundingInfo struct {
	// Surface selects the API surface the search tool rides: "" (provider
	// default), "responses", "chat_completions", or "messages".
	Surface string

	// ResponsesPath overrides the path joined onto BaseURL for the
	// responses surface (default "/responses").
	ResponsesPath string

	// ToolType is the tool's `type` value. Empty = provider default.
	ToolType string

	// ToolName is the tool's `name` value (messages surface only). Empty =
	// "web_search".
	ToolName string

	// MaxUses bounds how many searches one request may run. 0 = unbounded.
	MaxUses int
}

// Supports reports whether the model advertises a capability. A model with
// no capabilities declared supports exactly "chat".
func (m ModelInfo) Supports(capability string) bool {
	if len(m.Capabilities) == 0 {
		return capability == "chat"
	}
	for _, c := range m.Capabilities {
		if c == capability {
			return true
		}
	}
	return false
}

// ModelResolver resolves a logical model ID to its dispatch metadata. This
// is the domain's view of the model registry — the production adapter wraps
// internal/registry.
type ModelResolver interface {
	// Resolve returns the dispatch metadata for the given logical model ID.
	// An unknown ID fails the resolve (the caller decides whether to refuse
	// or fall back).
	Resolve(ctx context.Context, id LogicalModelID) (ModelInfo, error)
}
