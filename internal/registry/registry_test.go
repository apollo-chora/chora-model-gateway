package registry_test

import (
	"testing"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/registry"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func testSpecs() []registry.ModelSpec {
	return []registry.ModelSpec{
		{
			ID:            "gpt-4o",
			Provider:      "openai",
			UpstreamModel: "gpt-4o",
			BaseURL:       "https://api.openai.com/v1",
			Capabilities:  []string{"chat", "tools"},
		},
		{
			ID:            "chora-fast",
			Provider:      "openai",
			UpstreamModel: "gpt-4o-mini",
			BaseURL:       "https://api.openai.com/v1",
			Capabilities:  []string{"chat", "tools"},
			FallbackIDs:   []string{"gpt-4o"},
			Aliases:       []string{"fast", "quick"},
		},
		{
			ID:            "claude-sonnet-5",
			Provider:      "anthropic",
			UpstreamModel: "claude-sonnet-5",
			BaseURL:       "https://api.anthropic.com",
			Capabilities:  []string{"chat", "tools", "vision"},
		},
	}
}

func TestRegistryGet(t *testing.T) {
	reg, err := registry.Finalize(testSpecs())
	require.NoError(t, err)

	// By canonical id.
	spec, ok := reg.Get("gpt-4o")
	require.True(t, ok)
	assert.Equal(t, "gpt-4o", spec.ID)

	// Case-insensitive.
	spec, ok = reg.Get("GPT-4O")
	require.True(t, ok)
	assert.Equal(t, "gpt-4o", spec.ID)

	// By alias, case-insensitive.
	spec, ok = reg.Get("Fast")
	require.True(t, ok)
	assert.Equal(t, "chora-fast", spec.ID)

	// Unknown name.
	_, ok = reg.Get("ghost")
	assert.False(t, ok)

	// Empty name.
	_, ok = reg.Get("")
	assert.False(t, ok)
}

func TestRegistryList(t *testing.T) {
	reg, err := registry.Finalize(testSpecs())
	require.NoError(t, err)

	list := reg.List()
	require.Len(t, list, 3)
	// Sorted by ID; each entry appears once despite aliases.
	assert.Equal(t, "chora-fast", list[0].ID)
	assert.Equal(t, "claude-sonnet-5", list[1].ID)
	assert.Equal(t, "gpt-4o", list[2].ID)
}

func TestRegistryResolve(t *testing.T) {
	reg, err := registry.Finalize(testSpecs())
	require.NoError(t, err)

	t.Run("primary only", func(t *testing.T) {
		primary, chain, err := reg.Resolve("gpt-4o", nil)
		require.NoError(t, err)
		assert.Equal(t, "gpt-4o", primary.ID)
		assert.Empty(t, chain)
	})

	t.Run("registry fallback chain", func(t *testing.T) {
		primary, chain, err := reg.Resolve("chora-fast", nil)
		require.NoError(t, err)
		assert.Equal(t, "chora-fast", primary.ID)
		require.Len(t, chain, 1)
		assert.Equal(t, "gpt-4o", chain[0].ID)
	})

	t.Run("caller fallbacks come first", func(t *testing.T) {
		primary, chain, err := reg.Resolve("chora-fast", []string{"claude-sonnet-5"})
		require.NoError(t, err)
		assert.Equal(t, "chora-fast", primary.ID)
		require.Len(t, chain, 2)
		assert.Equal(t, "claude-sonnet-5", chain[0].ID)
		assert.Equal(t, "gpt-4o", chain[1].ID)
	})

	t.Run("dedupes the primary out of the chain", func(t *testing.T) {
		// The primary's own id as a caller fallback is dropped.
		_, chain, err := reg.Resolve("chora-fast", []string{"chora-fast"})
		require.NoError(t, err)
		require.Len(t, chain, 1)
		assert.Equal(t, "gpt-4o", chain[0].ID)
	})

	t.Run("dedupes repeated fallbacks", func(t *testing.T) {
		_, chain, err := reg.Resolve("gpt-4o", []string{"claude-sonnet-5", "claude-sonnet-5"})
		require.NoError(t, err)
		require.Len(t, chain, 1)
		assert.Equal(t, "claude-sonnet-5", chain[0].ID)
	})

	t.Run("unknown primary fails", func(t *testing.T) {
		_, _, err := reg.Resolve("ghost", nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), `"ghost" is not in the registry`)
	})

	t.Run("unknown caller fallback fails the whole resolve", func(t *testing.T) {
		_, _, err := reg.Resolve("gpt-4o", []string{"ghost"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), `"ghost" is not in the registry`)
	})

	t.Run("alias resolves to the same spec", func(t *testing.T) {
		primary, _, err := reg.Resolve("quick", nil)
		require.NoError(t, err)
		assert.Equal(t, "chora-fast", primary.ID)
	})
}

func TestFinalizeEmptyRegistry(t *testing.T) {
	_, err := registry.Finalize(nil)
	require.Error(t, err)
	var cfgErr *registry.ConfigError
	require.ErrorAs(t, err, &cfgErr)
	assert.Equal(t, "no models configured", cfgErr.Msg)
}

func TestFinalizeDuplicateDetection(t *testing.T) {
	t.Run("same name, different upstreams", func(t *testing.T) {
		specs := []registry.ModelSpec{
			{ID: "a", Provider: "openai", UpstreamModel: "x", BaseURL: "https://x.test"},
			{ID: "a", Provider: "openai", UpstreamModel: "y", BaseURL: "https://x.test"},
		}
		_, err := registry.Finalize(specs)
		require.Error(t, err)
		assert.Contains(t, err.Error(), `model name "a" is declared twice with different upstreams ("x")`)
	})

	t.Run("alias colliding with another id", func(t *testing.T) {
		specs := []registry.ModelSpec{
			{ID: "a", Provider: "openai", UpstreamModel: "x", BaseURL: "https://x.test"},
			{ID: "b", Provider: "openai", UpstreamModel: "y", BaseURL: "https://x.test", Aliases: []string{"a"}},
		}
		_, err := registry.Finalize(specs)
		require.Error(t, err)
		assert.Contains(t, err.Error(), `model name "a" is declared twice with different upstreams ("x")`)
	})

	t.Run("same name, same upstream — later wins", func(t *testing.T) {
		specs := []registry.ModelSpec{
			{ID: "a", Provider: "openai", UpstreamModel: "x", BaseURL: "https://first.test"},
			{ID: "a", Provider: "openai", UpstreamModel: "x", BaseURL: "https://second.test"},
		}
		reg, err := registry.Finalize(specs)
		require.NoError(t, err)
		spec, ok := reg.Get("a")
		require.True(t, ok)
		assert.Equal(t, "https://second.test", spec.BaseURL)
	})
}

func TestFinalizeFallbackExistence(t *testing.T) {
	specs := []registry.ModelSpec{
		{ID: "a", Provider: "openai", UpstreamModel: "x", BaseURL: "https://x.test", FallbackIDs: []string{"missing"}},
	}
	_, err := registry.Finalize(specs)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `model "a" lists fallback "missing", which is not in the registry`)
}
