package openai

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// A dimension refusal is a gateway configuration refusal, not an upstream
// fault: the request asked for a vector length the resolved model cannot
// produce, or the route returned one that contradicts its own configuration.
// It maps to the same OpenAI envelope as the other registry refusals, with the
// gateway-controlled message relayed.
func TestMapDimensionMismatchError_InvalidRequest(t *testing.T) {
	err := fmt.Errorf("embed: vendor dispatch: %w", &domain.DimensionMismatchError{
		Model:    "text-embedding-004",
		Expected: 1024,
		Actual:   768,
		Detail:   "embed: model \"text-embedding-004\" returned a 768-dimensional vector, expected 1024; refusing (never truncated or padded)",
	})
	got := mapEmbedError(err)
	if got.status != http.StatusBadRequest {
		t.Errorf("want 400, got %d", got.status)
	}
	if got.errType != "invalid_request_error" || got.code != "invalid_request" {
		t.Errorf("unexpected envelope %+v", got)
	}
	if got.message != err.Error() {
		t.Errorf("the gateway-controlled message must be relayed; got %q", got.message)
	}
}
