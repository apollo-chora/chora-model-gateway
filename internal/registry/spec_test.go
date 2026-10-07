package registry_test

import (
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/registry"
	"github.com/stretchr/testify/assert"
)

func TestModelSpecSupports(t *testing.T) {
	spec := registry.ModelSpec{Capabilities: []string{"chat", "tools", "vision"}}
	assert.True(t, spec.Supports("chat"))
	assert.True(t, spec.Supports("tools"))
	assert.True(t, spec.Supports("vision"))
	assert.False(t, spec.Supports("image"))
	assert.False(t, spec.Supports("embeddings"))
	assert.False(t, spec.Supports("web_search"))
}

func TestModelSpecSupportsEmptyCapabilitiesMeansChat(t *testing.T) {
	spec := registry.ModelSpec{}
	assert.True(t, spec.Supports("chat"))
	assert.False(t, spec.Supports("tools"))
	assert.False(t, spec.Supports("image"))
}

func TestModelSpecAPIKey(t *testing.T) {
	t.Setenv("REGISTRY_TEST_KEY", "secret-value")
	spec := registry.ModelSpec{APIKeyEnv: "REGISTRY_TEST_KEY"}
	assert.Equal(t, "secret-value", spec.APIKey())

	// An empty APIKeyEnv means no auth — never touches the environment.
	noAuth := registry.ModelSpec{}
	assert.Equal(t, "", noAuth.APIKey())

	// A set-but-empty env var resolves to empty (the dispatch layer treats
	// that as a misconfiguration, not as no-auth).
	t.Setenv("REGISTRY_TEST_EMPTY", "")
	empty := registry.ModelSpec{APIKeyEnv: "REGISTRY_TEST_EMPTY"}
	assert.Equal(t, "", empty.APIKey())
}

func TestModelSpecCostMicros(t *testing.T) {
	pricing := registry.Pricing{
		InputPerMtokUSDMicros:      2500000,  // $2.50/Mtok
		OutputPerMtokUSDMicros:     10000000, // $10.00/Mtok
		CachedPerMtokUSDMicros:     1250000,
		CacheWritePerMtokUSDMicros: 3125000,
	}
	// 2 Mtok input + 1 Mtok output = 2*2.50 + 1*10.00 = $15.00 = 15,000,000 micros.
	assert.Equal(t, int64(15_000_000), pricing.CostMicros(2_000_000, 1_000_000, 0, 0))
	// Cached + cache-write legs: 2 Mtok cached at $1.25/Mtok + 1 Mtok write
	// at $3.125/Mtok = $5.625.
	assert.Equal(t, int64(5_625_000), pricing.CostMicros(0, 0, 2_000_000, 1_000_000))
	// Term-wise floored: 1 token at $2.50/Mtok = 2.5 micros → 2.
	assert.Equal(t, int64(2), pricing.CostMicros(1, 0, 0, 0))
	// Unpriced table debits zero.
	assert.Equal(t, int64(0), registry.Pricing{}.CostMicros(1_000_000, 1_000_000, 0, 0))
}

func TestResolveEndpoint(t *testing.T) {
	tests := []struct {
		name        string
		base        string
		path        string
		defaultPath string
		want        string
	}{
		{"empty path uses default", "https://api.openai.com/v1", "", "/chat/completions", "https://api.openai.com/v1/chat/completions"},
		{"relative path joins base", "https://api.openai.com/v1", "/custom/chat", "/chat/completions", "https://api.openai.com/v1/custom/chat"},
		{"base trailing slash collapsed", "https://api.openai.com/v1/", "/chat/completions", "/chat/completions", "https://api.openai.com/v1/chat/completions"},
		{"path leading slash trimmed", "https://api.openai.com/v1", "chat/completions", "/chat/completions", "https://api.openai.com/v1/chat/completions"},
		{"absolute url wins verbatim", "https://api.openai.com/v1", "https://llm.internal.example.com/generate", "/chat/completions", "https://llm.internal.example.com/generate"},
		{"empty base returns path", "", "/chat/completions", "/chat/completions", "/chat/completions"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, registry.ResolveEndpoint(tt.base, tt.path, tt.defaultPath))
		})
	}
}

func TestModelSpecURLHelpers(t *testing.T) {
	spec := registry.ModelSpec{
		BaseURL:             "https://api.openai.com/v1",
		ChatCompletionsPath: "https://llm.internal.example.com/generate",
		MessagesPath:        "/v1/messages",
		ImagesPath:          "/images/generations",
		EmbeddingsPath:      "/embeddings",
		Grounding:           &registry.GroundingSpec{ResponsesPath: "/responses"},
	}
	assert.Equal(t, "https://llm.internal.example.com/generate", spec.ChatURL())
	assert.Equal(t, "https://api.openai.com/v1/v1/messages", spec.MessagesURL())
	assert.Equal(t, "https://api.openai.com/v1/images/generations", spec.ImagesURL())
	assert.Equal(t, "https://api.openai.com/v1/embeddings", spec.EmbeddingsURL())
	assert.Equal(t, "https://api.openai.com/v1/responses", spec.ResponsesURL())

	// Defaults when no overrides are declared.
	bare := registry.ModelSpec{BaseURL: "https://api.openai.com/v1"}
	assert.Equal(t, "https://api.openai.com/v1/chat/completions", bare.ChatURL())
	assert.Equal(t, "https://api.openai.com/v1/responses", bare.ResponsesURL())
}

func TestGroundingEffectiveDefaults(t *testing.T) {
	// Empty block: provider-family defaults.
	empty := registry.GroundingSpec{}
	assert.Equal(t, "responses", empty.EffectiveSurface("openai"))
	assert.Equal(t, "messages", empty.EffectiveSurface("anthropic"))
	assert.Equal(t, "web_search", empty.EffectiveToolType("responses"))
	assert.Equal(t, "web_search_20250305", empty.EffectiveToolType("messages"))
	assert.Equal(t, "web_search_preview", empty.EffectiveToolType("chat_completions"))
	assert.Equal(t, "web_search", empty.EffectiveToolName())

	// Explicit values win.
	explicit := registry.GroundingSpec{Surface: "chat_completions", ToolType: "web_search", ToolName: "search"}
	assert.Equal(t, "chat_completions", explicit.EffectiveSurface("anthropic"))
	assert.Equal(t, "web_search", explicit.EffectiveToolType("messages"))
	assert.Equal(t, "search", explicit.EffectiveToolName())
}

func TestModelSpecString(t *testing.T) {
	spec := registry.ModelSpec{Provider: "openai", UpstreamModel: "gpt-4o-mini"}
	assert.Equal(t, "openai:gpt-4o-mini", spec.String())
}

func TestKindFromCapabilities(t *testing.T) {
	tests := []struct {
		caps []string
		want registry.Kind
	}{
		{[]string{"chat", "tools"}, registry.KindText},
		{[]string{"image"}, registry.KindImage},
		{[]string{"embeddings"}, registry.KindEmbedding},
		// image + chat is a text model that can also see images.
		{[]string{"chat", "vision"}, registry.KindText},
		// image without chat wins over embeddings without chat.
		{[]string{"image", "embeddings"}, registry.KindImage},
		{[]string{"chat", "tools", "web_search"}, registry.KindText},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, registry.KindFromCapabilities(tt.caps), "caps=%v", tt.caps)
	}
}
