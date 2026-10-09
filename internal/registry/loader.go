package registry

import (
	"fmt"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// ConfigError is a registry/deployment misconfiguration. The loader returns
// it when the YAML or the role-based env config fails boot-time validation —
// a typo must fail at boot, not on the first request that touches it.
type ConfigError struct {
	Msg string
}

func (e *ConfigError) Error() string { return e.Msg }

// ----------------------------------------------------------------------------
// Load pipeline
// ----------------------------------------------------------------------------

// Load reads the registry YAML at path (when the file exists), layers the
// role-based env configuration on top, validates, and returns the registry.
// A missing file is not an error by itself — the env configuration may still
// provide models — but an unreadable one is.
func Load(path string) (Registry, error) {
	rows, err := readRegistryFile(path)
	if err != nil {
		return nil, err
	}
	specs, err := specsFromRows(rows)
	if err != nil {
		return nil, err
	}
	return Finalize(ApplyEnv(specs))
}

// ParseYAML parses + per-row validates registry YAML bytes. It does NOT
// layer the role-based env configuration (see ApplyEnv) and does NOT run
// the cross-entry checks (see Finalize).
func ParseYAML(data []byte) ([]ModelSpec, error) {
	rows, err := readRegistryFileBytes(data)
	if err != nil {
		return nil, err
	}
	return specsFromRows(rows)
}

// ApplyEnv layers the role-based environment configuration (TEXT_LLM_*,
// IMAGE_LLM_*, EMBEDDING_LLM_*) over the parsed specs and returns the merged
// slice. The role-based configuration is the primary deployment path: an
// env entry whose id collides with a YAML row wins when both declare the
// same upstream, and fails the duplicate check when they do not.
func ApplyEnv(specs []ModelSpec) []ModelSpec {
	merged := make([]ModelSpec, 0, len(specs)+len(envRoles))
	merged = append(merged, specs...)
	return append(merged, envRoleSpecs()...)
}

// Finalize runs the cross-entry validation (duplicate detection, fallback
// existence, empty registry) and builds the in-memory registry.
func Finalize(specs []ModelSpec) (Registry, error) {
	byKey := make(map[string]ModelSpec, len(specs))
	upstreams := make(map[string]string, len(specs)) // key → upstream model
	for _, spec := range specs {
		for _, name := range append([]string{spec.ID}, spec.Aliases...) {
			key := normalizeName(name)
			if key == "" {
				continue
			}
			if existing, ok := upstreams[key]; ok && existing != spec.UpstreamModel {
				return nil, &ConfigError{Msg: fmt.Sprintf(
					"model name %q is declared twice with different upstreams (%q)", name, existing)}
			}
			upstreams[key] = spec.UpstreamModel
			byKey[key] = spec
		}
	}

	for _, spec := range specs {
		for _, fb := range spec.FallbackIDs {
			if _, ok := byKey[normalizeName(fb)]; !ok {
				return nil, &ConfigError{Msg: fmt.Sprintf(
					"model %q lists fallback %q, which is not in the registry", spec.ID, fb)}
			}
		}
	}

	if len(byKey) == 0 {
		return nil, &ConfigError{Msg: "no models configured"}
	}

	// Canonical spec list: one entry per ID (last declaration wins, matching
	// byKey), sorted by ID for a deterministic listing.
	byID := make(map[string]ModelSpec, len(specs))
	for _, spec := range specs {
		byID[spec.ID] = spec
	}
	ids := make([]string, 0, len(byID))
	for id := range byID {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	canonical := make([]ModelSpec, 0, len(ids))
	for _, id := range ids {
		canonical = append(canonical, byID[id])
	}

	return &memoryRegistry{byKey: byKey, specs: canonical}, nil
}

// ----------------------------------------------------------------------------
// YAML reading
// ----------------------------------------------------------------------------

// registryFile is the on-disk shape: a top-level `models` list of mappings.
type registryFile struct {
	Models []map[string]any `yaml:"models"`
}

func readRegistryFile(path string) ([]map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read registry: %w", err)
	}
	return readRegistryFileBytes(data)
}

func readRegistryFileBytes(data []byte) ([]map[string]any, error) {
	var file registryFile
	if err := yaml.Unmarshal(data, &file); err != nil {
		return nil, &ConfigError{Msg: fmt.Sprintf("parse registry: %v", err)}
	}
	return file.Models, nil
}

// ----------------------------------------------------------------------------
// Per-row parsing + validation
// ----------------------------------------------------------------------------

// endpointPathFields are the four endpoint override fields, in validation
// order.
var endpointPathFields = []string{"chat_completions_path", "messages_path", "images_path", "embeddings_path"}

func specsFromRows(rows []map[string]any) ([]ModelSpec, error) {
	specs := make([]ModelSpec, 0, len(rows))
	for _, row := range rows {
		spec, err := specFromRow(row)
		if err != nil {
			return nil, err
		}
		specs = append(specs, spec)
	}
	return specs, nil
}

func specFromRow(row map[string]any) (ModelSpec, error) {
	if _, ok := row["id"]; !ok {
		return ModelSpec{}, &ConfigError{Msg: "id is required for a registry entry"}
	}
	id := yamlString(row, "id")

	caps, err := yamlStringSlice(row, "capabilities")
	if err != nil {
		return ModelSpec{}, err
	}
	if len(caps) == 0 {
		caps = []string{"chat"}
	}
	for _, cap := range caps {
		if !isKnownCapability(cap) {
			return ModelSpec{}, &ConfigError{Msg: fmt.Sprintf(
				"capability %q is not one of %s", cap, strings.Join(KnownCapabilities, ", "))}
		}
	}

	// Kind is derived from the capability list, so it is computed before the
	// base URL: a row that omits base_url inherits the role endpoint for its
	// own kind.
	kind := KindFromCapabilities(caps)

	provider := strings.ToLower(yamlString(row, "provider"))
	if !isKnownProvider(provider) {
		return ModelSpec{}, &ConfigError{Msg: fmt.Sprintf(
			"provider %q is not one of: %s", provider, strings.Join(KnownProviders, ", "))}
	}

	baseURL := resolveBaseURL(kind, yamlString(row, "base_url"))
	// SSRF/egress guard: the base URL must not point at a private/internal
	// IP. The gateway is the only caller of the vendor endpoints; a registry
	// entry that could reach an internal service would turn the gateway
	// into an internal-network proxy. An empty base URL is allowed here (a
	// self-hosted entry the deployment did not point at a role) — the
	// dispatch-time check refuses it rather than guessing an endpoint.
	if baseURL != "" {
		if err := domain.ValidateEgressURL(baseURL); err != nil {
			return ModelSpec{}, &ConfigError{Msg: fmt.Sprintf("base_url %q: %v", baseURL, err)}
		}
	}

	paths := make(map[string]string, len(endpointPathFields))
	for _, field := range endpointPathFields {
		path := yamlString(row, field)
		if err := validateEndpoint(field, path); err != nil {
			return ModelSpec{}, err
		}
		// SSRF/egress guard: an absolute endpoint override must also not
		// point at a private/internal IP.
		if path != "" && (strings.HasPrefix(path, "http://") || strings.HasPrefix(path, "https://")) {
			if err := domain.ValidateEgressURL(path); err != nil {
				return ModelSpec{}, &ConfigError{Msg: fmt.Sprintf("%s %q: %v", field, path, err)}
			}
		}
		paths[field] = path
	}

	grounding, err := parseGrounding(row)
	if err != nil {
		return ModelSpec{}, err
	}
	if grounding != nil && !hasCapability(caps, "web_search") {
		return ModelSpec{}, &ConfigError{Msg: "a grounding block is configured but `capabilities` does not include 'web_search'"}
	}

	contextWindow, err := yamlInt(row, "context_window")
	if err != nil {
		return ModelSpec{}, err
	}
	if contextWindow < 0 {
		return ModelSpec{}, &ConfigError{Msg: "context_window cannot be negative"}
	}
	maxOutput, err := yamlInt(row, "max_output_tokens")
	if err != nil {
		return ModelSpec{}, err
	}
	if maxOutput < 0 {
		return ModelSpec{}, &ConfigError{Msg: "max_output_tokens cannot be negative"}
	}
	if maxOutput > 0 && contextWindow > 0 && maxOutput > contextWindow {
		return ModelSpec{}, &ConfigError{Msg: fmt.Sprintf(
			"max_output_tokens (%d) exceeds context_window (%d)", maxOutput, contextWindow)}
	}

	upstreamModel := yamlString(row, "upstream_model")
	if upstreamModel == "" {
		upstreamModel = id
	}

	pricing, err := parsePricing(row)
	if err != nil {
		return ModelSpec{}, err
	}
	extraHeaders, err := parseExtraHeaders(row)
	if err != nil {
		return ModelSpec{}, err
	}
	fallbackIDs, err := yamlStringSlice(row, "fallback_ids")
	if err != nil {
		return ModelSpec{}, err
	}
	aliases, err := yamlStringSlice(row, "aliases")
	if err != nil {
		return ModelSpec{}, err
	}

	format := "chat_completions"
	if provider == "anthropic" {
		format = "messages"
	}

	return ModelSpec{
		ID:                  id,
		Kind:                kind,
		Provider:            provider,
		Format:              format,
		UpstreamModel:       upstreamModel,
		BaseURL:             baseURL,
		ChatCompletionsPath: paths["chat_completions_path"],
		MessagesPath:        paths["messages_path"],
		ImagesPath:          paths["images_path"],
		EmbeddingsPath:      paths["embeddings_path"],
		APIKeyEnv:           yamlString(row, "api_key_env"),
		Capabilities:        caps,
		FallbackIDs:         fallbackIDs,
		Aliases:             aliases,
		ContextWindow:       contextWindow,
		MaxOutputTokens:     maxOutput,
		Pricing:             pricing,
		ExtraHeaders:        extraHeaders,
		Grounding:           grounding,
	}, nil
}

// roleBaseURLEnv maps a registry Kind to the role-based env var that supplies
// its API root. The deployment registry (chora-stack/config/model-gateway/
// models.prod.yaml) declares no base_url on any row: the deployment supplies
// the endpoint through the role configuration, exactly as it supplies the
// credential through {PREFIX}_LLM_API_KEY. A row that omits base_url therefore
// inherits the endpoint for its own kind.
var roleBaseURLEnv = map[Kind]string{
	KindText:      "TEXT_LLM_BASE_URL",
	KindImage:     "IMAGE_LLM_BASE_URL",
	KindEmbedding: "EMBEDDING_LLM_BASE_URL",
}

// resolveBaseURL returns the entry's declared base_url, or — when the row
// omits it — the role-based env URL for the entry's kind. An empty result
// means the entry is self-hosted and no role endpoint is configured for it;
// the caller decides whether that is dispatchable.
func resolveBaseURL(kind Kind, declared string) string {
	if declared != "" {
		return declared
	}
	return strings.TrimSpace(os.Getenv(roleBaseURLEnv[kind]))
}

// KindFromCapabilities derives the dispatch shape from a capability list:
// an entry with `image` but no `chat` is an image model; one with
// `embeddings` but no `chat` is an embedding model; everything else is a
// text model.
func KindFromCapabilities(caps []string) Kind {
	switch {
	case hasCapability(caps, "image") && !hasCapability(caps, "chat"):
		return KindImage
	case hasCapability(caps, "embeddings") && !hasCapability(caps, "chat"):
		return KindEmbedding
	default:
		return KindText
	}
}

func hasCapability(caps []string, capability string) bool {
	for _, c := range caps {
		if c == capability {
			return true
		}
	}
	return false
}

func isKnownCapability(capability string) bool {
	return hasCapability(KnownCapabilities, capability)
}

func isKnownProvider(provider string) bool {
	return hasCapability(KnownProviders, provider)
}

// validateEndpoint enforces the two ways to point at a server: a relative
// path (joined onto base_url) or a valid absolute http(s) URL (used
// verbatim). A scheme-less //host/path or a non-http override is a
// configuration typo and must fail at boot, not at dispatch.
func validateEndpoint(field, path string) error {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil
	}
	if strings.Contains(path, "://") {
		lower := strings.ToLower(path)
		if !strings.HasPrefix(lower, "http://") && !strings.HasPrefix(lower, "https://") {
			return &ConfigError{Msg: fmt.Sprintf(
				"%s %q must be an http:// or https:// url to override base_url", field, path)}
		}
		u, err := url.Parse(path)
		if err != nil || u.Scheme == "" || u.Host == "" {
			return &ConfigError{Msg: fmt.Sprintf("%s %q is not a valid absolute url", field, path)}
		}
		return nil
	}
	if strings.HasPrefix(path, "//") {
		return &ConfigError{Msg: fmt.Sprintf(
			"%s %q looks like a url but is missing its scheme", field, path)}
	}
	return nil
}

// parseGrounding parses the optional `grounding:` block. A block without
// the web_search capability is rejected by the caller.
func parseGrounding(row map[string]any) (*GroundingSpec, error) {
	block, ok := row["grounding"]
	if !ok || block == nil {
		return nil, nil
	}
	m, ok := block.(map[string]any)
	if !ok {
		return nil, &ConfigError{Msg: "grounding must be a mapping"}
	}
	surface := strings.ToLower(yamlString(m, "surface"))
	switch surface {
	case "", "responses", "chat_completions", "messages":
	default:
		return nil, &ConfigError{Msg: fmt.Sprintf(
			"grounding surface %q is not one of: responses, chat_completions, messages", m["surface"])}
	}
	maxUses, err := yamlInt(m, "max_uses")
	if err != nil {
		return nil, err
	}
	if maxUses < 0 {
		return nil, &ConfigError{Msg: "grounding max_uses cannot be negative"}
	}
	extra, _ := m["extra_tool_fields"].(map[string]any)
	return &GroundingSpec{
		Surface:         surface,
		ResponsesPath:   yamlString(m, "responses_path"),
		ToolType:        yamlString(m, "tool_type"),
		ToolName:        yamlString(m, "tool_name"),
		MaxUses:         maxUses,
		ExtraToolFields: extra,
	}, nil
}

func parsePricing(row map[string]any) (Pricing, error) {
	v, ok := row["pricing"]
	if !ok || v == nil {
		return Pricing{}, nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		return Pricing{}, &ConfigError{Msg: "pricing must be a mapping"}
	}
	var p Pricing
	var err error
	if p.InputPerMtokUSDMicros, err = yamlInt64(m, "input_per_mtok_usd_micros"); err != nil {
		return Pricing{}, err
	}
	if p.OutputPerMtokUSDMicros, err = yamlInt64(m, "output_per_mtok_usd_micros"); err != nil {
		return Pricing{}, err
	}
	if p.CachedPerMtokUSDMicros, err = yamlInt64(m, "cached_per_mtok_usd_micros"); err != nil {
		return Pricing{}, err
	}
	if p.CacheWritePerMtokUSDMicros, err = yamlInt64(m, "cache_write_per_mtok_usd_micros"); err != nil {
		return Pricing{}, err
	}
	return p, nil
}

func parseExtraHeaders(row map[string]any) (map[string]string, error) {
	v, ok := row["extra_headers"]
	if !ok || v == nil {
		return map[string]string{}, nil
	}
	m, ok := v.(map[string]any)
	if !ok {
		return nil, &ConfigError{Msg: "extra_headers must be a mapping"}
	}
	out := make(map[string]string, len(m))
	for k, val := range m {
		out[fmt.Sprint(k)] = fmt.Sprint(val)
	}
	return out, nil
}

// ----------------------------------------------------------------------------
// Role-based env configuration
// ----------------------------------------------------------------------------

// envRole is one role-based env configuration source.
type envRole struct {
	prefix        string
	kind          Kind
	defaultFormat string
}

var envRoles = []envRole{
	{prefix: "TEXT", kind: KindText, defaultFormat: "responses"},
	{prefix: "IMAGE", kind: KindImage, defaultFormat: "images"},
	{prefix: "EMBEDDING", kind: KindEmbedding, defaultFormat: "embeddings"},
}

// envRoleSpecs builds the role-based env specs. A role activates when
// {PREFIX}_LLM_MODEL is set; the spec is constructed directly and skips the
// per-row YAML validation (the vendor is always openai/anthropic and the
// base URL is allowed to be empty for self-hosted servers).
func envRoleSpecs() []ModelSpec {
	specs := make([]ModelSpec, 0, len(envRoles))
	for _, role := range envRoles {
		if spec, ok := envRoleSpec(role); ok {
			specs = append(specs, spec)
		}
	}
	return specs
}

func envRoleSpec(role envRole) (ModelSpec, bool) {
	model := strings.TrimSpace(os.Getenv(role.prefix + "_LLM_MODEL"))
	if model == "" {
		return ModelSpec{}, false
	}
	grounding := envFlag(role.prefix + "_LLM_SUPPORT_GROUNDING")

	caps := roleCapabilities(role.kind)
	if role.kind == KindText && grounding {
		caps = append(caps, "web_search")
	}

	format := os.Getenv(role.prefix + "_LLM_FORMAT")
	if format == "" {
		format = role.defaultFormat
	}
	vendor := "openai"
	if format == "messages" {
		vendor = "anthropic"
	}

	baseURL := os.Getenv(role.prefix + "_LLM_BASE_URL")
	// SSRF/egress guard: the env-configured base URL must not point at a
	// private/internal IP. An empty base URL is allowed (self-hosted
	// servers may omit it); a non-empty one is validated.
	if baseURL != "" {
		if err := domain.ValidateEgressURL(baseURL); err != nil {
			// Fail loud: a misconfigured env base URL is a deployment
			// error, not a silent skip.
			panic(fmt.Sprintf("registry: %s_LLM_BASE_URL %q: %v", role.prefix, baseURL, err))
		}
	}

	spec := ModelSpec{
		ID:            model,
		Kind:          role.kind,
		Provider:      vendor,
		Format:        format,
		UpstreamModel: model,
		BaseURL:       baseURL,
		APIKeyEnv:     role.prefix + "_LLM_API_KEY",
		Capabilities:  caps,
	}
	if grounding {
		spec.Grounding = &GroundingSpec{}
	}
	return spec, true
}

func roleCapabilities(kind Kind) []string {
	switch kind {
	case KindImage:
		return []string{"image"}
	case KindEmbedding:
		return []string{"embeddings"}
	default:
		return []string{"chat", "tools"}
	}
}

// envFlag mirrors the Python truthiness set: the lowercased raw value must
// be exactly one of 1/true/yes/on (no trimming).
func envFlag(key string) bool {
	switch strings.ToLower(os.Getenv(key)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// ----------------------------------------------------------------------------
// YAML scalar coercion helpers
//
// The Python loader coerces row values defensively (str()/int() with
// defaults); these helpers mirror that so a YAML type quirk (a numeric
// string, a float) fails or coerces the same way at boot.
// ----------------------------------------------------------------------------

func yamlString(row map[string]any, key string) string {
	v, ok := row[key]
	if !ok || v == nil {
		return ""
	}
	return strings.TrimSpace(fmt.Sprint(v))
}

// yamlInt coerces a YAML value to int: ints pass through, floats truncate,
// numeric strings parse. Anything else is a boot-time ConfigError.
func yamlInt(row map[string]any, key string) (int, error) {
	n, err := yamlInt64(row, key)
	return int(n), err
}

func yamlInt64(row map[string]any, key string) (int64, error) {
	v, ok := row[key]
	if !ok || v == nil {
		return 0, nil
	}
	switch n := v.(type) {
	case int:
		return int64(n), nil
	case int64:
		return n, nil
	case float64:
		return int64(n), nil
	case string:
		parsed, err := strconv.ParseInt(strings.TrimSpace(n), 10, 64)
		if err != nil {
			return 0, &ConfigError{Msg: fmt.Sprintf("%s %q is not a valid integer", key, n)}
		}
		return parsed, nil
	default:
		return 0, &ConfigError{Msg: fmt.Sprintf("%s is not a valid integer", key)}
	}
}

// yamlStringSlice coerces a YAML list of scalars to []string. A missing or
// null value yields nil; a non-list value is a boot-time ConfigError (the
// Python loader would iterate a scalar into nonsense and fail later anyway).
func yamlStringSlice(row map[string]any, key string) ([]string, error) {
	v, ok := row[key]
	if !ok || v == nil {
		return nil, nil
	}
	items, ok := v.([]any)
	if !ok {
		return nil, &ConfigError{Msg: fmt.Sprintf("%s must be a list", key)}
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		out = append(out, fmt.Sprint(item))
	}
	return out, nil
}
