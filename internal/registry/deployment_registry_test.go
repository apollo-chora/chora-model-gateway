package registry_test

import (
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/registry"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// deploymentRegistryPath is the registry the apollo deployment actually
// mounts (chora-stack/compose.prod.yml: CHORA_MODEL_REGISTRY). It lives in a
// sibling repo, so the test skips when the checkout is not present — but when
// it IS present, "the Go gateway can load the deployment registry" is a real
// cross-repo invariant, not a shape we re-typed from memory.
func deploymentRegistryPath(t *testing.T) string {
	t.Helper()
	candidate := filepath.Join("..", "..", "..", "chora-stack", "config", "model-gateway", "models.prod.yaml")
	if _, err := os.Stat(candidate); err != nil {
		t.Skipf("deployment registry not present at %s: %v", candidate, err)
	}
	return candidate
}

// The deployment registry declares its endpoints on the rows themselves; the
// role-based env URLs are the fallback for rows that declare none (e.g. the
// self-hosted and stub entries). The loader rejects a row with neither.
func TestLoadDeploymentRegistry(t *testing.T) {
	clearEnvRoleVars(t)
	t.Setenv("TEXT_LLM_BASE_URL", "https://api.meta.ai/v1")
	t.Setenv("IMAGE_LLM_BASE_URL", "https://api.meta.ai/v1")
	t.Setenv("EMBEDDING_LLM_BASE_URL", "https://openrouter.ai/api/v1")

	reg, err := registry.Load(deploymentRegistryPath(t))
	require.NoError(t, err)

	// The deployment's embedding route: the logical id consumers pin
	// (text-embedding-004) resolves to LiquidAI's embedding model on
	// OpenRouter, NOT to a Google publisher model.
	embed, ok := reg.Get("text-embedding-004")
	require.True(t, ok, "the deployment registry must still declare the pinned embedding route")
	assert.Equal(t, "liquid/lfm-2.5-embedding-350m:free", embed.UpstreamModel)
	assert.Equal(t, "openrouter.ai", hostOf(embed.BaseURL))
	assert.Equal(t, "https://openrouter.ai/api/v1/embeddings", embed.EmbeddingsURL())
	assert.True(t, embed.Supports("embeddings"))
	assert.Empty(t, embed.FallbackIDs, "the pinned route must have no fallbacks")

	// The deployment's text route: every non-grounded text caller resolves to
	// LongCat-2.5-Preview. The row declares its endpoint explicitly; the
	// role-based env URL remains the fallback for rows that declare none.
	text, ok := reg.Get("longcat-2.5-preview")
	require.True(t, ok)
	assert.Equal(t, "LongCat-2.5-Preview", text.UpstreamModel)
	assert.Equal(t, "https://api.longcat.ai/openai/v1", text.BaseURL)
	assert.True(t, text.Supports("chat"))
}

// hostOf extracts the host of an absolute http(s) URL, lowercased. Returns ""
// for anything that does not parse.
func hostOf(raw string) string {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return ""
	}
	return strings.ToLower(u.Hostname())
}
