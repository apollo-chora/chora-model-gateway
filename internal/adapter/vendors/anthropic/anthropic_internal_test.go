package anthropic

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type stubSecrets struct{}

func (stubSecrets) ResolveByoaKey(ctx context.Context, tenantID string, vendor domain.VendorFamily) (string, error) {
	return "k", nil
}

func (stubSecrets) ResolveGlobalKey(ctx context.Context, vendor domain.VendorFamily) (string, error) {
	return "", nil
}

func TestNew_DefaultClientIsPooledWith300sTimeout(t *testing.T) {
	c, err := New(Config{Secrets: stubSecrets{}})
	require.NoError(t, err)
	// The default client carries the 300s timeout (the Python transport's
	// ceiling — a grounded reasoning run can take well over a minute) on a
	// connection-pooled transport.
	assert.Equal(t, 300*time.Second, c.httpClient.Timeout)
	transport, ok := c.httpClient.Transport.(*http.Transport)
	require.True(t, ok, "default transport must be *http.Transport, got %T", c.httpClient.Transport)
	assert.Equal(t, 100, transport.MaxIdleConns)
	assert.Equal(t, 100, transport.MaxIdleConnsPerHost)
	assert.Equal(t, 90*time.Second, transport.IdleConnTimeout)
}

func TestNew_InjectedClientKeptVerbatim(t *testing.T) {
	injected := &http.Client{Timeout: 5 * time.Second}
	c, err := New(Config{HTTPClient: injected, Secrets: stubSecrets{}})
	require.NoError(t, err)
	assert.Same(t, injected, c.httpClient)
}

func TestNew_CustomTimeoutHonored(t *testing.T) {
	c, err := New(Config{Secrets: stubSecrets{}, Timeout: 42 * time.Second})
	require.NoError(t, err)
	assert.Equal(t, 42*time.Second, c.httpClient.Timeout)
}

func TestAnthropicCostMicros(t *testing.T) {
	// claude-opus-4-7: 15_000_000 input / 75_000_000 output / 1_500_000 cached /
	// 18_750_000 cache-write, per 1M tokens.
	assert.Equal(t, int64(0), anthropicCostMicros("claude-opus-4-7", 0, 0, 0, 0))
	// 1M plain input, no cache: 15_000_000.
	assert.Equal(t, int64(15_000_000), anthropicCostMicros("claude-opus-4-7", 1_000_000, 0, 0, 0))
	// 1M cache reads: 1_500_000.
	assert.Equal(t, int64(1_500_000), anthropicCostMicros("claude-opus-4-7", 0, 0, 1_000_000, 0))
	// 1M cache writes: 18_750_000.
	assert.Equal(t, int64(18_750_000), anthropicCostMicros("claude-opus-4-7", 0, 0, 0, 1_000_000))
	// 1M output: 75_000_000.
	assert.Equal(t, int64(75_000_000), anthropicCostMicros("claude-opus-4-7", 0, 1_000_000, 0, 0))
	// Cached tokens beyond the plain input clamp the BILLABLE input at 0 (the
	// gemini adapter's convention — never credit negative fresh input) while
	// the cached term still bills at the cache rate: 999 * 1_500_000/1e6.
	assert.Equal(t, int64(1_498), anthropicCostMicros("claude-opus-4-7", 10, 0, 999, 0))
	// Unknown model: zero cost.
	assert.Equal(t, int64(0), anthropicCostMicros("claude-vapor-X", 1_000_000, 1_000_000, 0, 0))
}

func TestGroundingConfig_Tool(t *testing.T) {
	// Defaults.
	g := &GroundingConfig{}
	tool := g.tool()
	assert.Equal(t, map[string]any{"type": "web_search_20250305", "name": "web_search"}, tool)

	// MaxUses omitted at 0, extra fields merged, "type" excluded.
	g = &GroundingConfig{
		ToolType:    "web_search_20250305",
		ToolName:    "search",
		MaxUses:     2,
		ExtraFields: map[string]any{"type": "bogus", "cache_control": map[string]any{"type": "ephemeral"}},
	}
	tool = g.tool()
	assert.Equal(t, map[string]any{
		"type":          "web_search_20250305",
		"name":          "search",
		"max_uses":      2,
		"cache_control": map[string]any{"type": "ephemeral"},
	}, tool)
}

func TestTruncate(t *testing.T) {
	assert.Equal(t, "", truncate("", 10))
	assert.Equal(t, "short", truncate("short", 10))
	assert.Equal(t, "1234567890", truncate("1234567890", 10))
	assert.Equal(t, "1234567890…", truncate("12345678901", 10))
}

func TestResolveEndpoint(t *testing.T) {
	assert.Equal(t, "https://api.anthropic.com/v1/messages", resolveEndpoint("https://api.anthropic.com", "", "/v1/messages"))
	assert.Equal(t, "https://api.anthropic.com/proxy/messages", resolveEndpoint("https://api.anthropic.com", "/proxy/messages", "/v1/messages"))
	assert.Equal(t, "https://api.anthropic.com/proxy/messages", resolveEndpoint("https://api.anthropic.com/", "proxy/messages", "/v1/messages"))
	assert.Equal(t, "https://other.example/absolute", resolveEndpoint("https://api.anthropic.com", "https://other.example/absolute", "/v1/messages"))
	assert.Equal(t, "/v1/messages", resolveEndpoint("", "", "/v1/messages"))
}
