// Package registry is the YAML-based model registry for the model gateway.
// Ported from the Python reference (chora-model-gateway-python/app/config.py +
// config/models.yaml): the registry file IS the catalogue — adding a model is
// an edit to the YAML, not a code change.
//
// The package is standalone: it carries no domain, gRPC, or cloud imports so
// it can be consumed by any adapter (HTTP, gRPC, dispatch) without pulling
// the rest of the gateway along.
package registry

import (
	"fmt"
	"os"
	"strings"
)

// KnownCapabilities enumerates the capability tokens a registry entry may
// advertise. Anything else fails boot-time validation.
var KnownCapabilities = []string{"chat", "tools", "vision", "image", "embeddings", "web_search"}

// KnownProviders enumerates the provider tokens the registry accepts.
var KnownProviders = []string{"openai", "anthropic"}

// Kind classifies a registry entry by its dispatch shape. Derived from the
// capability list at load time: an entry with `image` but no `chat` is an
// image model; one with `embeddings` but no `chat` is an embedding model;
// everything else is a text model.
type Kind string

const (
	KindText      Kind = "text"
	KindImage     Kind = "image"
	KindEmbedding Kind = "embedding"
)

// Format names the wire shape the provider speaks. Anthropic entries are
// "messages"; OpenAI entries are "chat_completions". Role-based env entries
// may also carry the surface name ("responses", "images", "embeddings").
type Format string

// Pricing holds the USD micros-per-million-tokens rate table. A zero value
// means unpriced — the ledger records the usage and debits zero.
type Pricing struct {
	InputPerMtokUSDMicros      int64 `yaml:"input_per_mtok_usd_micros"`
	OutputPerMtokUSDMicros     int64 `yaml:"output_per_mtok_usd_micros"`
	CachedPerMtokUSDMicros     int64 `yaml:"cached_per_mtok_usd_micros"`
	CacheWritePerMtokUSDMicros int64 `yaml:"cache_write_per_mtok_usd_micros"`
}

// CostMicros returns the USD micros for one call, term-wise floored — the
// same arithmetic the ledger stores. cached/cacheWrites default to 0.
func (p Pricing) CostMicros(billableInput, output, cached, cacheWrites int64) int64 {
	micros := billableInput * p.InputPerMtokUSDMicros / 1_000_000
	micros += cached * p.CachedPerMtokUSDMicros / 1_000_000
	micros += cacheWrites * p.CacheWritePerMtokUSDMicros / 1_000_000
	micros += output * p.OutputPerMtokUSDMicros / 1_000_000
	return micros
}

// GroundingSpec is the hosted web search configuration for one registry
// entry. Grounding is OPTIONAL and per-entry: a model without a Grounding
// block refuses a grounded request rather than silently running an
// ungrounded call that would only LOOK researched.
type GroundingSpec struct {
	// Surface selects the API surface the search tool rides: "" (provider
	// default), "responses", "chat_completions", or "messages".
	Surface string `yaml:"surface"`

	// ResponsesPath overrides the path joined onto BaseURL for the
	// responses surface (default "/responses").
	ResponsesPath string `yaml:"responses_path"`

	// ToolType is the tool's `type` value. Empty = provider default for the
	// surface.
	ToolType string `yaml:"tool_type"`

	// ToolName is the tool's `name` value (messages surface only). Empty =
	// "web_search".
	ToolName string `yaml:"tool_name"`

	// MaxUses bounds how many searches one request may run. 0 = unbounded.
	// Load-bearing for Anthropic, which prices web search PER SEARCH.
	MaxUses int `yaml:"max_uses"`

	// ExtraToolFields are provider tool options the gateway does not model,
	// merged verbatim into the tool object (except "type", which is owned
	// by ToolType).
	ExtraToolFields map[string]any `yaml:"extra_tool_fields,omitempty"`
}

// EffectiveSurface returns the surface to ground on: the explicit Surface,
// or the provider-family default ("messages" for anthropic, else
// "responses").
func (g GroundingSpec) EffectiveSurface(vendor string) string {
	if g.Surface != "" {
		return g.Surface
	}
	if vendor == "anthropic" {
		return "messages"
	}
	return "responses"
}

// EffectiveToolType returns the tool `type` for the surface: the explicit
// ToolType, or the surface default ("web_search_20250305" for messages,
// "web_search_preview" for chat_completions, else "web_search").
func (g GroundingSpec) EffectiveToolType(surface string) string {
	if g.ToolType != "" {
		return g.ToolType
	}
	switch surface {
	case "messages":
		return "web_search_20250305"
	case "chat_completions":
		return "web_search_preview"
	default:
		return "web_search"
	}
}

// EffectiveToolName returns the tool `name` (messages surface): the explicit
// ToolName, or "web_search".
func (g GroundingSpec) EffectiveToolName() string {
	if g.ToolName != "" {
		return g.ToolName
	}
	return "web_search"
}

// ModelSpec is one registry entry: everything the gateway needs to dispatch
// a call against one model — where it lives, what it can do, what it costs,
// and what to try when it fails.
type ModelSpec struct {
	// ID is the name callers pass as `model`. Lookups are case-insensitive;
	// the field itself keeps the declared casing.
	ID string `yaml:"id"`

	// Kind is the dispatch shape, derived from Capabilities at load time.
	Kind Kind `yaml:"kind"`

	// Provider is the adapter that speaks to it: "openai" or "anthropic".
	Provider string `yaml:"provider"`

	// Format is the wire shape: "chat_completions" (OpenAI) or "messages"
	// (Anthropic). Role-based env entries may carry a surface name instead.
	Format string `yaml:"format"`

	// UpstreamModel is the model name sent to the provider. Defaults to ID.
	UpstreamModel string `yaml:"upstream_model"`

	// BaseURL is the API root. Optional: a row may omit it when the deployment
	// supplies the endpoint through the role-based env config
	// ({TEXT,IMAGE,EMBEDDING}_LLM_BASE_URL) — see resolveBaseURL. An empty
	// BaseURL is a self-hosted entry with no role endpoint configured.
	BaseURL string `yaml:"base_url"`

	// Endpoint path overrides. Empty means the provider default
	// (/chat/completions, /v1/messages, /images/generations, /embeddings);
	// an absolute http(s) URL is used verbatim and BaseURL is ignored for
	// that surface.
	ChatCompletionsPath string `yaml:"chat_completions_path"`
	MessagesPath        string `yaml:"messages_path"`
	ImagesPath          string `yaml:"images_path"`
	EmbeddingsPath      string `yaml:"embeddings_path"`

	// APIKeyEnv is the env var holding the credential. Empty = no auth.
	APIKeyEnv string `yaml:"api_key_env"`

	// Capabilities advertises what the model can do. Empty means ["chat"].
	Capabilities []string `yaml:"capabilities,omitempty"`

	// FallbackIDs are tried in order when this model fails.
	FallbackIDs []string `yaml:"fallback_ids,omitempty"`

	// Aliases are extra names this entry answers to.
	Aliases []string `yaml:"aliases,omitempty"`

	// ContextWindow is the total tokens (input + output) the model accepts.
	ContextWindow int `yaml:"context_window"`

	// MaxOutputTokens is the generation ceiling; the gateway clamps callers
	// to it. 0 = unbounded.
	MaxOutputTokens int `yaml:"max_output_tokens"`

	// Pricing is the USD micros per million tokens rate table.
	Pricing Pricing `yaml:"pricing"`

	// ExtraHeaders are provider-specific headers sent on every dispatch.
	ExtraHeaders map[string]string `yaml:"extra_headers,omitempty"`

	// Grounding is the hosted web search configuration. Nil = the model
	// refuses grounded requests.
	Grounding *GroundingSpec `yaml:"grounding,omitempty"`
}

// Supports reports whether the entry advertises a capability. An entry with
// no capabilities declared supports exactly "chat".
func (s ModelSpec) Supports(capability string) bool {
	if len(s.Capabilities) == 0 {
		return capability == "chat"
	}
	for _, c := range s.Capabilities {
		if c == capability {
			return true
		}
	}
	return false
}

// APIKey resolves the credential from the environment. An empty APIKeyEnv
// means the entry needs no auth and returns "".
func (s ModelSpec) APIKey() string {
	if s.APIKeyEnv == "" {
		return ""
	}
	return strings.TrimSpace(os.Getenv(s.APIKeyEnv))
}

// CostMicros returns the USD micros for one call against this entry's price
// table, term-wise floored.
func (s ModelSpec) CostMicros(billableInput, output, cached, cacheWrites int64) int64 {
	return s.Pricing.CostMicros(billableInput, output, cached, cacheWrites)
}

// ResolveEndpoint joins a path override onto a base URL. An empty path falls
// back to defaultPath; an absolute http(s) URL is used verbatim (base is
// ignored); anything else is a relative path joined onto the base.
func ResolveEndpoint(base, path, defaultPath string) string {
	if path == "" {
		path = defaultPath
	}
	if strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://") {
		return path
	}
	if base == "" {
		return path
	}
	return strings.TrimRight(base, "/") + "/" + strings.TrimLeft(path, "/")
}

// ChatURL resolves the chat-completions endpoint for this entry.
func (s ModelSpec) ChatURL() string {
	return ResolveEndpoint(s.BaseURL, s.ChatCompletionsPath, "/chat/completions")
}

// ResponsesURL resolves the responses endpoint (hosted web search) for this
// entry. The path override lives on the grounding block.
func (s ModelSpec) ResponsesURL() string {
	path := ""
	if s.Grounding != nil {
		path = s.Grounding.ResponsesPath
	}
	return ResolveEndpoint(s.BaseURL, path, "/responses")
}

// MessagesURL resolves the Anthropic messages endpoint for this entry.
func (s ModelSpec) MessagesURL() string {
	return ResolveEndpoint(s.BaseURL, s.MessagesPath, "/v1/messages")
}

// ImagesURL resolves the image-generation endpoint for this entry.
func (s ModelSpec) ImagesURL() string {
	return ResolveEndpoint(s.BaseURL, s.ImagesPath, "/images/generations")
}

// EmbeddingsURL resolves the embeddings endpoint for this entry.
func (s ModelSpec) EmbeddingsURL() string {
	return ResolveEndpoint(s.BaseURL, s.EmbeddingsPath, "/embeddings")
}

// String renders the entry as "provider:upstream" for logs and the fallback
// chain record.
func (s ModelSpec) String() string {
	return fmt.Sprintf("%s:%s", s.Provider, s.UpstreamModel)
}
