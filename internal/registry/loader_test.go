package registry_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/registry"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writeRegistry writes a registry YAML to a temp file and returns its path.
func writeRegistry(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "models.yaml")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	return path
}

// clearEnvRoleVars blanks the role-based env configuration so a test starts
// from a known-empty env layer. t.Setenv with "" mirrors "unset" because the
// env loader treats an empty model name as "role not configured".
func clearEnvRoleVars(t *testing.T) {
	t.Helper()
	for _, prefix := range []string{"TEXT", "IMAGE", "EMBEDDING"} {
		t.Setenv(prefix+"_LLM_MODEL", "")
		t.Setenv(prefix+"_LLM_FORMAT", "")
		t.Setenv(prefix+"_LLM_BASE_URL", "")
		t.Setenv(prefix+"_LLM_API_KEY", "")
		t.Setenv(prefix+"_LLM_SUPPORT_GROUNDING", "")
	}
}

func TestLoadTestdataRegistry(t *testing.T) {
	clearEnvRoleVars(t)
	reg, err := registry.Load("testdata/models.yaml")
	require.NoError(t, err)

	list := reg.List()
	require.Len(t, list, 10)

	// Spot-check the entries that exercise the interesting paths.
	spec, ok := reg.Get("gpt-4o")
	require.True(t, ok)
	assert.Equal(t, registry.KindText, spec.Kind)
	assert.Equal(t, "chat_completions", spec.Format)
	assert.Equal(t, 128000, spec.ContextWindow)
	assert.Equal(t, 16384, spec.MaxOutputTokens)
	assert.Equal(t, "OPENAI_API_KEY", spec.APIKeyEnv)
	assert.True(t, spec.Supports("embeddings"))
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", spec.ChatURL())
	assert.Equal(t, int64(15_000_000), spec.CostMicros(2_000_000, 1_000_000, 0, 0))

	// Fallback chain + aliases.
	fast, ok := reg.Get("chora-fast")
	require.True(t, ok)
	assert.Equal(t, "gpt-4o-mini", fast.UpstreamModel)
	_, chain, err := reg.Resolve("fast", nil) // alias lookup
	require.NoError(t, err)
	require.Len(t, chain, 1)
	assert.Equal(t, "gpt-4o", chain[0].ID)

	// Image entry: kind derived from capabilities, zero pricing.
	image, ok := reg.Get("gpt-image-1")
	require.True(t, ok)
	assert.Equal(t, registry.KindImage, image.Kind)
	assert.True(t, image.Supports("image"))
	assert.False(t, image.Supports("chat"))

	// Self-hosted: no api_key_env, extra headers preserved.
	local, ok := reg.Get("local-llama")
	require.True(t, ok)
	assert.Equal(t, "", local.APIKeyEnv)
	assert.Equal(t, "chora", local.ExtraHeaders["X-Tenant"])

	// Absolute endpoint override wins verbatim.
	internal, ok := reg.Get("internal-llm")
	require.True(t, ok)
	assert.Equal(t, "https://llm.internal.example.com/generate", internal.ChatURL())

	// Anthropic: messages format.
	claude, ok := reg.Get("claude-sonnet-5")
	require.True(t, ok)
	assert.Equal(t, "messages", claude.Format)
	assert.Equal(t, "https://api.anthropic.com/v1/messages", claude.MessagesURL())

	// Grounding blocks.
	search, ok := reg.Get("gpt-4o-search")
	require.True(t, ok)
	require.NotNil(t, search.Grounding)
	assert.Equal(t, "responses", search.Grounding.Surface)
	assert.Equal(t, "https://api.openai.com/v1/responses", search.ResponsesURL())

	claudeSearch, ok := reg.Get("claude-sonnet-5-search")
	require.True(t, ok)
	require.NotNil(t, claudeSearch.Grounding)
	assert.Equal(t, "messages", claudeSearch.Grounding.Surface)
	assert.Equal(t, 5, claudeSearch.Grounding.MaxUses)
	assert.Equal(t, "web_search_20250305", claudeSearch.Grounding.EffectiveToolType("messages"))

	// Embedding entry.
	embed, ok := reg.Get("text-embedding-004")
	require.True(t, ok)
	assert.Equal(t, registry.KindEmbedding, embed.Kind)
	assert.Equal(t, "https://api.openai.com/v1/embeddings", embed.EmbeddingsURL())
}

func TestLoadMissingFileFallsBackToEnv(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_MODEL", "env-model")

	reg, err := registry.Load(filepath.Join(t.TempDir(), "does-not-exist.yaml"))
	require.NoError(t, err)
	spec, ok := reg.Get("env-model")
	require.True(t, ok)
	assert.Equal(t, "env-model", spec.UpstreamModel)
}

func TestLoadMissingFileNoEnvIsEmptyRegistry(t *testing.T) {
	clearEnvRoleVars(t)
	_, err := registry.Load(filepath.Join(t.TempDir(), "does-not-exist.yaml"))
	require.Error(t, err)
	var cfgErr *registry.ConfigError
	require.ErrorAs(t, err, &cfgErr)
	assert.Equal(t, "no models configured", cfgErr.Msg)
}

func TestLoadUnreadableFileFails(t *testing.T) {
	clearEnvRoleVars(t)
	// A directory cannot be read as a file.
	_, err := registry.Load(t.TempDir())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "read registry")
}

// ----------------------------------------------------------------------------
// Boot-time validation
// ----------------------------------------------------------------------------

func TestLoadRejectsUnknownCapability(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    capabilities: [chat, telepathy]
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `capability "telepathy" is not one of chat, tools, vision, image, embeddings, web_search`)
}

func TestLoadRejectsUnknownProvider(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openrouter
    base_url: https://x.test/v1
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `provider "openrouter" is not one of: openai, anthropic`)
}

func TestLoadRejectsMissingProvider(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    base_url: https://x.test/v1
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `provider "" is not one of: openai, anthropic`)
}

// A row may omit base_url when the deployment supplies the endpoint through
// the role-based env config. The deployment registry
// (chora-stack/config/model-gateway/models.prod.yaml) declares no base_url on
// any row, so this must load — refusing it is what stopped the Go gateway
// from booting on the deployment registry.
func TestLoadAcceptsMissingBaseURLWithRoleBaseURL(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_BASE_URL", "https://role.test/v1")
	t.Setenv("EMBEDDING_LLM_BASE_URL", "https://openrouter.ai/api/v1")

	path := writeRegistry(t, `
models:
  - id: text-model
    provider: openai
    capabilities: [chat]
  - id: embed-model
    provider: openai
    capabilities: [embeddings]
`)
	reg, err := registry.Load(path)
	require.NoError(t, err)

	text, ok := reg.Get("text-model")
	require.True(t, ok)
	assert.Equal(t, "https://role.test/v1", text.BaseURL)
	assert.Equal(t, "https://role.test/v1/chat/completions", text.ChatURL())

	embed, ok := reg.Get("embed-model")
	require.True(t, ok)
	assert.Equal(t, "https://openrouter.ai/api/v1", embed.BaseURL)
	assert.Equal(t, "https://openrouter.ai/api/v1/embeddings", embed.EmbeddingsURL())
}

// The inherited role endpoint is validated exactly like a declared one: a
// deployment that points a role at an internal address still fails boot.
func TestLoadRejectsUnresolvableRoleBaseURL(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("EMBEDDING_LLM_BASE_URL", "http://127.0.0.1:8080")

	path := writeRegistry(t, `
models:
  - id: embed-model
    provider: openai
    capabilities: [embeddings]
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "private/internal ip")
}

// A row with no base_url and no role endpoint is a self-hosted entry. It
// loads (the env-role path has always allowed an empty base URL); the
// dispatch-time check is what refuses to guess an endpoint for it.
func TestLoadAcceptsMissingBaseURLWithoutRoleBaseURL(t *testing.T) {
	clearEnvRoleVars(t)

	path := writeRegistry(t, `
models:
  - id: self-hosted
    provider: openai
    upstream_model: llama3.1:8b
    capabilities: [chat]
`)
	reg, err := registry.Load(path)
	require.NoError(t, err)
	spec, ok := reg.Get("self-hosted")
	require.True(t, ok)
	assert.Equal(t, "", spec.BaseURL)
}

// A declared base_url always wins over the role endpoint.
func TestLoadDeclaredBaseURLWinsOverRole(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_BASE_URL", "https://role.test/v1")

	path := writeRegistry(t, `
models:
  - id: text-model
    provider: openai
    base_url: https://declared.test/v1
`)
	reg, err := registry.Load(path)
	require.NoError(t, err)
	spec, ok := reg.Get("text-model")
	require.True(t, ok)
	assert.Equal(t, "https://declared.test/v1", spec.BaseURL)
}

func TestLoadRejectsMissingID(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - provider: openai
    base_url: https://x.test/v1
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "id is required")
}

func TestLoadRejectsInvalidEndpointOverrides(t *testing.T) {
	clearEnvRoleVars(t)
	for _, bad := range []string{"//host/path", "ftp://x.test/v1", "grpc://x.test"} {
		path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    chat_completions_path: "`+bad+`"
`)
		_, err := registry.Load(path)
		require.Error(t, err, "bad path %q", bad)
		assert.Contains(t, err.Error(), "chat_completions_path")
	}
}

func TestLoadAcceptsValidEndpointOverrides(t *testing.T) {
	clearEnvRoleVars(t)
	for _, good := range []string{"/custom/chat", "https://x.test/v1/custom/chat", ""} {
		path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    chat_completions_path: "`+good+`"
`)
		reg, err := registry.Load(path)
		require.NoError(t, err, "good path %q", good)
		spec, ok := reg.Get("a")
		require.True(t, ok)
		assert.Equal(t, good, spec.ChatCompletionsPath)
	}
}

func TestLoadRejectsUnknownFallback(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    fallback_ids: [missing]
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `model "a" lists fallback "missing", which is not in the registry`)
}

func TestLoadRejectsMaxOutputExceedingContextWindow(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    context_window: 1000
    max_output_tokens: 2000
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max_output_tokens (2000) exceeds context_window (1000)")
}

func TestLoadRejectsNegativeContextWindow(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    context_window: -1
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "context_window cannot be negative")
}

func TestLoadRejectsNegativeMaxOutput(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    max_output_tokens: -5
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max_output_tokens cannot be negative")
}

func TestLoadAllowsMaxOutputExceedingZeroContextWindow(t *testing.T) {
	// The consistency check only fires when BOTH are positive: an image
	// entry (context_window: 0) may declare any output ceiling.
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    capabilities: [image]
    max_output_tokens: 1024
`)
	_, err := registry.Load(path)
	require.NoError(t, err)
}

func TestLoadRejectsDuplicateNameWithDifferentUpstreams(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    upstream_model: x
    base_url: https://x.test/v1
  - id: a
    provider: openai
    upstream_model: y
    base_url: https://x.test/v1
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `model name "a" is declared twice with different upstreams ("x")`)
}

func TestLoadRejectsGroundingWithoutWebSearchCapability(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    capabilities: [chat]
    grounding:
      surface: responses
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "grounding block is configured but `capabilities` does not include 'web_search'")
}

func TestLoadRejectsBadGroundingSurface(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    capabilities: [chat, web_search]
    grounding:
      surface: carrier_pigeon
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "grounding surface")
}

func TestLoadRejectsNegativeGroundingMaxUses(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    capabilities: [chat, web_search]
    grounding:
      max_uses: -1
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "grounding max_uses cannot be negative")
}

func TestLoadEmptyModelsListIsEmptyRegistry(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, "models: []\n")
	_, err := registry.Load(path)
	require.Error(t, err)
	var cfgErr *registry.ConfigError
	require.ErrorAs(t, err, &cfgErr)
	assert.Equal(t, "no models configured", cfgErr.Msg)
}

func TestLoadEmptyCapabilitiesDefaultsToChat(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://x.test/v1
    capabilities: []
`)
	reg, err := registry.Load(path)
	require.NoError(t, err)
	spec, ok := reg.Get("a")
	require.True(t, ok)
	assert.Equal(t, []string{"chat"}, spec.Capabilities)
	assert.True(t, spec.Supports("chat"))
	assert.False(t, spec.Supports("tools"))
}

func TestLoadDefaultsUpstreamModelToID(t *testing.T) {
	clearEnvRoleVars(t)
	path := writeRegistry(t, `
models:
  - id: my-model
    provider: openai
    base_url: https://x.test/v1
`)
	reg, err := registry.Load(path)
	require.NoError(t, err)
	spec, ok := reg.Get("my-model")
	require.True(t, ok)
	assert.Equal(t, "my-model", spec.UpstreamModel)
}

// ----------------------------------------------------------------------------
// Role-based env configuration
// ----------------------------------------------------------------------------

func TestEnvTextRole(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_MODEL", "gpt-4o")
	t.Setenv("TEXT_LLM_BASE_URL", "https://api.openai.com/v1")
	t.Setenv("TEXT_LLM_API_KEY", "OPENAI_API_KEY")

	reg, err := registry.Load(filepath.Join(t.TempDir(), "missing.yaml"))
	require.NoError(t, err)
	spec, ok := reg.Get("gpt-4o")
	require.True(t, ok)
	assert.Equal(t, registry.KindText, spec.Kind)
	assert.Equal(t, "openai", spec.Provider)
	assert.Equal(t, "responses", spec.Format) // TEXT default format
	assert.Equal(t, "https://api.openai.com/v1", spec.BaseURL)
	assert.Equal(t, "TEXT_LLM_API_KEY", spec.APIKeyEnv)
	assert.Equal(t, []string{"chat", "tools"}, spec.Capabilities)
	assert.Nil(t, spec.Grounding)
}

func TestEnvTextRoleGrounding(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_MODEL", "gpt-4o-search")
	t.Setenv("TEXT_LLM_SUPPORT_GROUNDING", "true")

	reg, err := registry.Load(filepath.Join(t.TempDir(), "missing.yaml"))
	require.NoError(t, err)
	spec, ok := reg.Get("gpt-4o-search")
	require.True(t, ok)
	assert.Equal(t, []string{"chat", "tools", "web_search"}, spec.Capabilities)
	require.NotNil(t, spec.Grounding)
	// Empty grounding block: provider-family defaults.
	assert.Equal(t, "responses", spec.Grounding.EffectiveSurface("openai"))
}

func TestEnvTextRoleGroundingFlagVariants(t *testing.T) {
	for _, truthy := range []string{"1", "true", "TRUE", "yes", "on"} {
		clearEnvRoleVars(t)
		t.Setenv("TEXT_LLM_MODEL", "m")
		t.Setenv("TEXT_LLM_SUPPORT_GROUNDING", truthy)
		reg, err := registry.Load(filepath.Join(t.TempDir(), "missing.yaml"))
		require.NoError(t, err)
		spec, ok := reg.Get("m")
		require.True(t, ok, "flag %q", truthy)
		assert.True(t, spec.Supports("web_search"), "flag %q", truthy)
	}
	for _, falsy := range []string{"0", "false", "no", "off", " true", ""} {
		clearEnvRoleVars(t)
		t.Setenv("TEXT_LLM_MODEL", "m")
		t.Setenv("TEXT_LLM_SUPPORT_GROUNDING", falsy)
		reg, err := registry.Load(filepath.Join(t.TempDir(), "missing.yaml"))
		require.NoError(t, err)
		spec, ok := reg.Get("m")
		require.True(t, ok, "flag %q", falsy)
		assert.False(t, spec.Supports("web_search"), "flag %q", falsy)
	}
}

func TestEnvFormatSelectsVendor(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_MODEL", "claude-sonnet-5")
	t.Setenv("TEXT_LLM_FORMAT", "messages")

	reg, err := registry.Load(filepath.Join(t.TempDir(), "missing.yaml"))
	require.NoError(t, err)
	spec, ok := reg.Get("claude-sonnet-5")
	require.True(t, ok)
	assert.Equal(t, "anthropic", spec.Provider)
	assert.Equal(t, "messages", spec.Format)
}

func TestEnvImageRole(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("IMAGE_LLM_MODEL", "gpt-image-1")

	reg, err := registry.Load(filepath.Join(t.TempDir(), "missing.yaml"))
	require.NoError(t, err)
	spec, ok := reg.Get("gpt-image-1")
	require.True(t, ok)
	assert.Equal(t, registry.KindImage, spec.Kind)
	assert.Equal(t, "openai", spec.Provider)
	assert.Equal(t, "images", spec.Format) // IMAGE default format
	assert.Equal(t, []string{"image"}, spec.Capabilities)
}

func TestEnvEmbeddingRole(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("EMBEDDING_LLM_MODEL", "text-embedding-004")

	reg, err := registry.Load(filepath.Join(t.TempDir(), "missing.yaml"))
	require.NoError(t, err)
	spec, ok := reg.Get("text-embedding-004")
	require.True(t, ok)
	assert.Equal(t, registry.KindEmbedding, spec.Kind)
	assert.Equal(t, "embeddings", spec.Format) // EMBEDDING default format
	assert.Equal(t, []string{"embeddings"}, spec.Capabilities)
}

func TestEnvRoleWinsOverYAMLRow(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_MODEL", "a") // same id + same upstream as the YAML row

	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    base_url: https://yaml.test/v1
`)
	reg, err := registry.Load(path)
	require.NoError(t, err)
	spec, ok := reg.Get("a")
	require.True(t, ok)
	// The env spec replaced the YAML row (same upstream, env layered last).
	assert.Equal(t, "responses", spec.Format)
	assert.Equal(t, "TEXT_LLM_API_KEY", spec.APIKeyEnv)
}

func TestEnvRoleCollidingUpstreamFails(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_MODEL", "a") // env upstream = "a", YAML upstream = "x"

	path := writeRegistry(t, `
models:
  - id: a
    provider: openai
    upstream_model: x
    base_url: https://yaml.test/v1
`)
	_, err := registry.Load(path)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `model name "a" is declared twice with different upstreams ("x")`)
}

func TestEnvSpecsSkipYAMLValidation(t *testing.T) {
	// An env role needs no base_url and no provider declaration — the spec
	// is constructed directly, so these must not fail boot.
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_MODEL", "self-hosted")
	t.Setenv("TEXT_LLM_BASE_URL", "")

	reg, err := registry.Load(filepath.Join(t.TempDir(), "missing.yaml"))
	require.NoError(t, err)
	_, ok := reg.Get("self-hosted")
	assert.True(t, ok)
}

// ----------------------------------------------------------------------------
// ParseYAML / ApplyEnv units
// ----------------------------------------------------------------------------

func TestParseYAMLSkipsEnvLayering(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_MODEL", "env-model")

	specs, err := registry.ParseYAML([]byte(`
models:
  - id: yaml-model
    provider: openai
    base_url: https://x.test/v1
`))
	require.NoError(t, err)
	require.Len(t, specs, 1)
	assert.Equal(t, "yaml-model", specs[0].ID)
}

func TestApplyEnvMergesRoles(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_MODEL", "text-model")
	t.Setenv("IMAGE_LLM_MODEL", "image-model")
	t.Setenv("EMBEDDING_LLM_MODEL", "embed-model")

	merged := registry.ApplyEnv([]registry.ModelSpec{{ID: "yaml-model"}})
	require.Len(t, merged, 4)
	assert.Equal(t, "yaml-model", merged[0].ID)
	assert.Equal(t, "text-model", merged[1].ID)
	assert.Equal(t, "image-model", merged[2].ID)
	assert.Equal(t, "embed-model", merged[3].ID)
}

func TestApplyEnvWithoutRolesKeepsSpecs(t *testing.T) {
	clearEnvRoleVars(t)
	merged := registry.ApplyEnv([]registry.ModelSpec{{ID: "yaml-model"}})
	require.Len(t, merged, 1)
}
