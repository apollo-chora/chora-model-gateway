package openai

import (
	"net/http"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// The per-tenant concurrency ceiling (CHORA_LLM_MAX_CONCURRENT_PER_TENANT) must
// surface as an OpenAI 429 rate_limit_error on both HTTP dispatch paths: the
// request never reached a provider, so it is retryable.
func TestMapOverloadedError_RateLimitOnEveryPath(t *testing.T) {
	err := &domain.OverloadedError{TenantID: "tenant-1", Limit: 1}
	cases := map[string]mappedError{
		"invoke": mapInvokeError(err),
		"embed":  mapEmbedError(err),
	}
	for name, got := range cases {
		if got.status != http.StatusTooManyRequests {
			t.Errorf("%s: want 429, got %d", name, got.status)
		}
		if got.errType != "rate_limit_error" || got.code != "tenant_concurrency_limit" {
			t.Errorf("%s: unexpected envelope %+v", name, got)
		}
	}
}
