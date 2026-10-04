// Package config loads the gateway's runtime configuration and the model
// registry.
//
// Two distinct concerns live here on purpose, because both are "what does
// this deployment look like" and neither belongs in the domain:
//
//   - Config: process-level settings (ports, database DSN, timeouts) read
//     from environment variables.
//   - Registry: the model catalogue — which logical model id maps to which
//     provider, which upstream model name, which endpoint, which
//     credential, which capabilities and limits. Read from a YAML file.
//
// NO value in this package is compiled into the binary. Endpoints, model
// names and API-key references all come from the environment or the
// registry file, so pointing the gateway at a different inference server is
// a config edit, never a code edit.
package config

import (
	"errors"
	"fmt"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// ---------------------------------------------------------------------------
// Process config
// ---------------------------------------------------------------------------

// Config is the process-level runtime configuration. Every field is
// environment-sourced; nothing here has a hard-coded production value.
type Config struct {
	// GRPCPort is the ModelGatewayService listen port (canonical 9090).
	GRPCPort int

	// HTTPPort serves both the health probes and the OpenAI-compatible
	// /v1 surface.
	HTTPPort int

	// DatabaseURL is the Postgres DSN for the budget + ledger tables.
	DatabaseURL string

	// BootstrapTimeout bounds how long boot waits for the database.
	BootstrapTimeout time.Duration

	// VendorHTTPTimeout bounds a single upstream provider call. Generous by
	// default: a large multimodal prompt on a local model server can sit for
	// a long time before the first byte.
	VendorHTTPTimeout time.Duration

	// RegistryPath is the model registry YAML file.
	RegistryPath string

	// DefaultTenantID is the tenant every request is attributed to when the
	// caller does not name one. Required: the budget row is RLS-scoped by
	// tenant, so a request without one cannot be settled.
	DefaultTenantID string

	// DefaultGCID is the actor every request is attributed to when the caller
	// does not name one.
	DefaultGCID string

	// DefaultAgentID is recorded on the ledger row for calls that arrive
	// without an agent (the OpenAI-compatible surface, by default).
	DefaultAgentID string

	// RequireAPIKey refuses a call whose target has no credential. On by
	// default: silently sending an unauthenticated request to a public
	// endpoint is worse than a clear refusal.
	RequireAPIKey bool

	// ServiceVersion is the build identifier stamped on every ledger row.
	ServiceVersion string
}

// LoadConfig reads the process configuration from the environment. It fails
// loud on a missing required value rather than substituting a default that
// would silently mis-attribute cost.
func LoadConfig() (*Config, error) {
	cfg := &Config{
		GRPCPort:        envInt("CHORA_GRPC_PORT", 9090),
		HTTPPort:        envInt("CHORA_HTTP_PORT", 8080),
		DatabaseURL:     strings.TrimSpace(os.Getenv("CHORA_DATABASE_URL")),
		RegistryPath:    envOr("CHORA_MODEL_REGISTRY", "config/models.yaml"),
		DefaultTenantID: strings.TrimSpace(os.Getenv("CHORA_DEFAULT_TENANT_ID")),
		DefaultGCID:     strings.TrimSpace(os.Getenv("CHORA_DEFAULT_GCID")),
		DefaultAgentID:  envOr("CHORA_DEFAULT_AGENT_ID", "openai_compat"),
		RequireAPIKey:   envBool("CHORA_REQUIRE_API_KEY", true),
		ServiceVersion:  envOr("SERVICE_VERSION", "chora-model-gateway:local"),
	}
	if cfg.DatabaseURL == "" {
		return nil, errors.New("config: CHORA_DATABASE_URL required (the budget + ledger tables live in Postgres)")
	}
	if cfg.DefaultTenantID == "" {
		return nil, errors.New("config: CHORA_DEFAULT_TENANT_ID required (the budget row is tenant-scoped)")
	}
	if cfg.DefaultGCID == "" {
		cfg.DefaultGCID = cfg.DefaultTenantID
	}

	var err error
	if cfg.BootstrapTimeout, err = envDuration("CHORA_BOOTSTRAP_TIMEOUT", 30*time.Second); err != nil {
		return nil, err
	}
	if cfg.VendorHTTPTimeout, err = envDuration("CHORA_VENDOR_HTTP_TIMEOUT", 120*time.Second); err != nil {
		return nil, err
	}
	return cfg, nil
}

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

// Pricing is the per-model cost table, in USD micros per million tokens —
// the same unit the ledger stores, so no conversion happens at write time.
// A zero entry means "unpriced": the call is recorded and budgeted at zero,
// which is the honest answer for a local model.
type Pricing struct {
	InputPerMTokUSDMicros  int64 `yaml:"input_per_mtok_usd_micros"`
	OutputPerMTokUSDMicros int64 `yaml:"output_per_mtok_usd_micros"`
	// CachedPerMTokUSDMicros prices cache READS. Providers discount these, so
	// leaving it zero while setting the input rate overcharges every cached
	// request.
	CachedPerMTokUSDMicros int64 `yaml:"cached_per_mtok_usd_micros"`
	// CacheWritePerMTokUSDMicros prices cache WRITES, which providers charge
	// at a PREMIUM over plain input. Zero here means cache writes are billed
	// as free, which under-bills a prompt-caching workload — set it if you use
	// prompt caching with this model.
	CacheWritePerMTokUSDMicros int64 `yaml:"cache_write_per_mtok_usd_micros"`
}

// ModelEntry is one row of the registry: everything needed to turn a
// caller-supplied model name into a concrete upstream request.
type ModelEntry struct {
	// ID is the name callers pass as `model` / logical_model_id. Required.
	ID string `yaml:"id"`

	// Provider selects the adapter: "openai" or "anthropic". Required.
	Provider string `yaml:"provider"`

	// UpstreamModel is the model name sent to the provider. Defaults to ID.
	UpstreamModel string `yaml:"upstream_model"`

	// BaseURL is the API root, e.g. "https://api.openai.com/v1". Required.
	BaseURL string `yaml:"base_url"`

	// ChatCompletionsPath is appended to BaseURL for text/tool calls.
	// Defaults to "/chat/completions". Set it to an ABSOLUTE url to point at
	// a fully custom chat-completions endpoint that ignores BaseURL.
	ChatCompletionsPath string `yaml:"chat_completions_path"`

	// MessagesPath is the Anthropic-shaped equivalent of
	// ChatCompletionsPath. Defaults to "/v1/messages". Only the Anthropic
	// adapter reads it.
	MessagesPath string `yaml:"messages_path"`

	// ImagesPath is appended to BaseURL for image generation. Defaults to
	// "/images/generations". Absolute values are used verbatim.
	ImagesPath string `yaml:"images_path"`

	// EmbeddingsPath is appended to BaseURL for embeddings. Defaults to
	// "/embeddings". Absolute values are used verbatim.
	EmbeddingsPath string `yaml:"embeddings_path"`

	// APIKeyEnv names the environment variable holding this model's
	// credential. Leave empty for a server that needs no key (vLLM, Ollama,
	// LM Studio).
	APIKeyEnv string `yaml:"api_key_env"`

	// ContextWindow is the model's total context in tokens. Zero = unknown.
	ContextWindow int `yaml:"context_window"`

	// MaxOutputTokens caps generation. Zero = uncapped.
	MaxOutputTokens int `yaml:"max_output_tokens"`

	// Capabilities gates which surfaces may address this model. Recognised:
	// chat, tools, vision, image, embeddings. Defaults to ["chat"].
	Capabilities []string `yaml:"capabilities"`

	// Pricing is the cost table used for the budget debit and the ledger.
	Pricing Pricing `yaml:"pricing"`

	// ExtraHeaders are provider-specific headers sent on every dispatch.
	ExtraHeaders map[string]string `yaml:"extra_headers"`

	// FallbackIDs is the ordered chain tried when this model fails. Each id
	// must resolve to another registry entry, so the chain may cross
	// providers.
	FallbackIDs []string `yaml:"fallback_ids"`

	// Aliases are extra names this entry answers to, so one upstream model can
	// be reachable under several names.
	Aliases []string `yaml:"aliases"`

	// Grounding configures hosted web search for this entry. OPTIONAL: leave it
	// out and a grounded request is refused, even if the entry lists the
	// web_search capability.
	//
	// Every field is optional, and the defaults follow the provider family:
	// an `openai` entry grounds on /v1/responses with {"type":"web_search"},
	// an `anthropic` entry on /v1/messages with
	// {"type":"web_search_20250305","name":"web_search"}.
	Grounding *GroundingEntry `yaml:"grounding"`
}

// GroundingEntry is the registry shape of domain.Grounding. Kept separate so a
// YAML typo fails with a config error rather than a confusing zero value.
type GroundingEntry struct {
	// Surface selects the endpoint: responses | chat_completions | messages.
	// Empty picks by provider family.
	Surface string `yaml:"surface"`

	// ResponsesPath is the Responses endpoint. Default "/responses".
	ResponsesPath string `yaml:"responses_path"`

	// ToolType is the tool discriminator. Empty picks the surface default
	// (web_search / web_search_preview / web_search_20250305).
	ToolType string `yaml:"tool_type"`

	// ToolName is Anthropic's required `name` field. Default "web_search".
	ToolName string `yaml:"tool_name"`

	// MaxUses caps searches per call. Zero = provider default. Anthropic
	// prices per search, so this is a spend control there.
	MaxUses int `yaml:"max_uses"`

	// ExtraToolFields are merged into the tool object verbatim, for provider
	// options this schema does not model (domain filters, user location,
	// search_context_size).
	ExtraToolFields map[string]any `yaml:"extra_tool_fields"`
}

// Registry is the parsed model catalogue.
type Registry struct {
	entries map[string]domain.TargetModel
	// pricing is kept alongside so the cost calculators can read it; the
	// domain TargetModel deliberately carries no pricing.
	pricing map[string]Pricing
	// fallbacks maps a model name to its declared chain of fallback names.
	fallbacks map[string][]string
	// order preserves declaration order so GET /v1/models is stable.
	order []string
}

// LoadRegistry parses and validates the registry YAML at path.
//
// Validation is strict on purpose: a registry with a typo in a provider
// name or an unparseable endpoint should fail at boot, not on the first
// request that happens to touch it.
func LoadRegistry(path string) (*Registry, error) {
	raw, err := os.ReadFile(path) // #nosec G304 -- path is operator-supplied config, read-only by design
	if err != nil {
		return nil, fmt.Errorf("config: read registry %s: %w", path, err)
	}
	var file struct {
		Models []ModelEntry `yaml:"models"`
	}
	if err := yaml.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("config: parse registry %s: %w", path, err)
	}
	if len(file.Models) == 0 {
		return nil, fmt.Errorf("config: registry %s declares no models", path)
	}

	reg := &Registry{
		entries:   make(map[string]domain.TargetModel, len(file.Models)),
		pricing:   make(map[string]Pricing, len(file.Models)),
		fallbacks: make(map[string][]string, len(file.Models)),
	}

	// Pass 1 — resolve each entry into a domain target, keyed by id + aliases.
	for _, e := range file.Models {
		target, err := e.toTarget()
		if err != nil {
			return nil, fmt.Errorf("config: model %q: %w", e.ID, err)
		}
		chain := make([]string, 0, len(e.FallbackIDs))
		for _, fb := range e.FallbackIDs {
			if trimmed := strings.ToLower(strings.TrimSpace(fb)); trimmed != "" {
				chain = append(chain, trimmed)
			}
		}
		for _, name := range append([]string{e.ID}, e.Aliases...) {
			key := strings.ToLower(strings.TrimSpace(name))
			if key == "" {
				continue
			}
			if existing, dup := reg.entries[key]; dup && existing.UpstreamModel != target.UpstreamModel {
				return nil, fmt.Errorf("config: model name %q is declared twice with different upstreams (%q)", name, existing.UpstreamModel)
			}
			t := target
			t.LogicalModelID = domain.LogicalModelID(key)
			reg.entries[key] = t
			reg.pricing[key] = e.Pricing
			reg.fallbacks[key] = chain
			if !contains(reg.order, key) {
				reg.order = append(reg.order, key)
			}
		}
	}

	// Pass 2 — validate the fallback chains now that every name resolves.
	for _, e := range file.Models {
		for _, fb := range e.FallbackIDs {
			if _, ok := reg.entries[strings.ToLower(strings.TrimSpace(fb))]; !ok {
				return nil, fmt.Errorf("config: model %q lists fallback %q, which is not in the registry", e.ID, fb)
			}
		}
	}
	return reg, nil
}

func (e ModelEntry) toTarget() (domain.TargetModel, error) {
	if strings.TrimSpace(e.ID) == "" {
		return domain.TargetModel{}, errors.New("id required")
	}
	family, err := parseProvider(e.Provider)
	if err != nil {
		return domain.TargetModel{}, err
	}
	upstream := strings.TrimSpace(e.UpstreamModel)
	if upstream == "" {
		upstream = strings.TrimSpace(e.ID)
	}
	if strings.TrimSpace(e.BaseURL) == "" {
		return domain.TargetModel{}, errors.New("base_url required")
	}
	if err := validateEndpoint(e); err != nil {
		return domain.TargetModel{}, err
	}

	caps := e.Capabilities
	if len(caps) == 0 {
		caps = []string{domain.CapabilityChat}
	}
	var grounding *domain.Grounding
	for _, c := range caps {
		if !contains(domain.KnownCapabilities, c) {
			return domain.TargetModel{}, fmt.Errorf("capability %q is not one of %s", c, strings.Join(domain.KnownCapabilities, ", "))
		}
	}
	if e.Grounding != nil {
		g, err := e.Grounding.toDomain()
		if err != nil {
			return domain.TargetModel{}, fmt.Errorf("grounding: %w", err)
		}
		// Declaring the tool without the capability would silently accept the
		// configuration and then refuse every grounded call at dispatch.
		if !slices.Contains(caps, domain.CapabilityWebSearch) {
			return domain.TargetModel{}, fmt.Errorf(
				"a grounding block is configured but `capabilities` does not include %q",
				domain.CapabilityWebSearch)
		}
		grounding = g
	}
	if e.ContextWindow < 0 {
		return domain.TargetModel{}, errors.New("context_window cannot be negative")
	}
	if e.MaxOutputTokens < 0 {
		return domain.TargetModel{}, errors.New("max_output_tokens cannot be negative")
	}
	if e.MaxOutputTokens > 0 && e.ContextWindow > 0 && e.MaxOutputTokens > e.ContextWindow {
		return domain.TargetModel{}, fmt.Errorf("max_output_tokens (%d) exceeds context_window (%d)", e.MaxOutputTokens, e.ContextWindow)
	}

	return domain.TargetModel{
		Vendor:              family,
		LogicalModelID:      domain.LogicalModelID(strings.ToLower(strings.TrimSpace(e.ID))),
		UpstreamModel:       upstream,
		BaseURL:             strings.TrimSpace(e.BaseURL),
		ChatCompletionsPath: strings.TrimSpace(e.ChatCompletionsPath),
		MessagesPath:        strings.TrimSpace(e.MessagesPath),
		ImagesPath:          strings.TrimSpace(e.ImagesPath),
		EmbeddingsPath:      strings.TrimSpace(e.EmbeddingsPath),
		APIKeyRef:           strings.TrimSpace(e.APIKeyEnv),
		ContextWindow:       e.ContextWindow,
		MaxOutputTokens:     e.MaxOutputTokens,
		Capabilities:        caps,
		ExtraHeaders:        e.ExtraHeaders,
		Grounding:           grounding,
	}, nil
}

// toDomain validates and converts one grounding block.
func (g GroundingEntry) toDomain() (*domain.Grounding, error) {
	out := &domain.Grounding{
		ResponsesPath:   strings.TrimSpace(g.ResponsesPath),
		ToolType:        strings.TrimSpace(g.ToolType),
		ToolName:        strings.TrimSpace(g.ToolName),
		MaxUses:         g.MaxUses,
		ExtraToolFields: g.ExtraToolFields,
	}
	switch strings.ToLower(strings.TrimSpace(g.Surface)) {
	case "":
		out.Surface = ""
	case string(domain.SurfaceResponses):
		out.Surface = domain.SurfaceResponses
	case string(domain.SurfaceChatCompletions):
		out.Surface = domain.SurfaceChatCompletions
	case string(domain.SurfaceMessages):
		out.Surface = domain.SurfaceMessages
	default:
		return nil, fmt.Errorf("surface %q is not one of: responses, chat_completions, messages", g.Surface)
	}
	if g.MaxUses < 0 {
		return nil, errors.New("max_uses cannot be negative")
	}
	return out, nil
}

// validateEndpoint checks that a custom path is either relative (joined onto
// base_url) or an absolute URL (used verbatim). A bare path with a scheme-
// less, host-bearing string is the common mistake this catches.
func validateEndpoint(e ModelEntry) error {
	for field, p := range map[string]string{
		"chat_completions_path": e.ChatCompletionsPath,
		"messages_path":         e.MessagesPath,
		"images_path":           e.ImagesPath,
		"embeddings_path":       e.EmbeddingsPath,
	} {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if strings.Contains(p, "://") {
			// Only http/https count as absolute overrides, because that is
			// exactly what domain.resolveEndpoint recognises. Accepting, say,
			// a grpc:// url here would pass boot validation and then be
			// silently concatenated onto base_url, producing a nonsense URL
			// that only fails at dispatch time — far from the config error.
			lower := strings.ToLower(p)
			if !strings.HasPrefix(lower, "http://") && !strings.HasPrefix(lower, "https://") {
				return fmt.Errorf("%s %q must be an http:// or https:// url to override base_url", field, p)
			}
			u, err := url.Parse(p)
			if err != nil || u.Host == "" {
				return fmt.Errorf("%s %q is not a valid absolute url", field, p)
			}
			continue
		}
		if strings.HasPrefix(p, "//") {
			return fmt.Errorf("%s %q looks like a url but is missing its scheme", field, p)
		}
	}
	return nil
}

func parseProvider(p string) (domain.VendorFamily, error) {
	switch strings.ToLower(strings.TrimSpace(p)) {
	case "openai":
		return domain.VendorFamilyOpenAI, nil
	case "anthropic":
		return domain.VendorFamilyAnthropic, nil
	case "":
		return "", errors.New("provider required")
	default:
		return "", fmt.Errorf("provider %q is not one of: openai, anthropic", p)
	}
}

// Lookup resolves a caller-supplied model name to its target.
func (r *Registry) Lookup(id domain.LogicalModelID) (domain.TargetModel, bool) {
	t, ok := r.entries[strings.ToLower(strings.TrimSpace(string(id)))]
	return t, ok
}

// Known reports whether a caller-supplied model name names a registry entry.
// The HTTP facade checks this before dispatch so an unknown name is a 404
// naming the fix, rather than a 502 from whichever provider it tried.
func (r *Registry) Known(id string) bool {
	_, ok := r.Lookup(domain.LogicalModelID(id))
	return ok
}

// PricingFor returns the cost table for a caller-supplied model name.
func (r *Registry) PricingFor(id domain.LogicalModelID) (Pricing, bool) {
	p, ok := r.pricing[strings.ToLower(strings.TrimSpace(string(id)))]
	return p, ok
}

// FallbacksFor returns the resolved fallback targets for a model id, in
// declaration order. Unknown ids are skipped rather than erroring here — the
// primary resolve has already succeeded by the time this is consulted.
func (r *Registry) FallbacksFor(id domain.LogicalModelID) []domain.TargetModel {
	key := strings.ToLower(strings.TrimSpace(string(id)))
	var out []domain.TargetModel
	for _, name := range r.fallbacks[key] {
		if t, ok := r.entries[name]; ok {
			out = append(out, t)
		}
	}
	return out
}

// Names returns every callable model name, in declaration order.
func (r *Registry) Names() []string {
	out := make([]string, len(r.order))
	copy(out, r.order)
	return out
}

// Targets returns every distinct target in declaration order.
func (r *Registry) Targets() []domain.TargetModel {
	out := make([]domain.TargetModel, 0, len(r.order))
	for _, name := range r.order {
		out = append(out, r.entries[name])
	}
	return out
}

// ---------------------------------------------------------------------------
// Small env helpers
// ---------------------------------------------------------------------------

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		var n int
		if _, err := fmt.Sscanf(v, "%d", &n); err == nil {
			return n
		}
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(key))) {
	case "true", "1", "yes", "on":
		return true
	case "false", "0", "no", "off":
		return false
	default:
		return fallback
	}
}

func envDuration(key string, fallback time.Duration) (time.Duration, error) {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback, nil
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return 0, fmt.Errorf("config: %s invalid: %w", key, err)
	}
	if d <= 0 {
		return 0, fmt.Errorf("config: %s must be positive, got %s", key, d)
	}
	return d, nil
}

func contains(haystack []string, needle string) bool {
	for _, h := range haystack {
		if h == needle {
			return true
		}
	}
	return false
}
