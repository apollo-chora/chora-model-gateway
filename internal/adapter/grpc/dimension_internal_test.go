package modelgatewaygrpc

import (
	"fmt"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// A dimension refusal is a gateway configuration refusal, not an upstream
// fault: the request asked for a vector length the resolved model cannot
// produce, or the route returned one that contradicts its own configuration.
// It maps to FAILED_PRECONDITION like the other registry refusals — nothing
// was billed, and the caller cannot fix it by retrying.
func TestMapDimensionMismatchError_FailedPrecondition(t *testing.T) {
	err := fmt.Errorf("embed: vendor dispatch: %w", &domain.DimensionMismatchError{
		Model:    "text-embedding-004",
		Expected: 1024,
		Actual:   768,
		Detail:   "embed: model \"text-embedding-004\" returned a 768-dimensional vector, expected 1024; refusing (never truncated or padded)",
	})
	got := mapEmbedError(err)
	if code := status.Code(got); code != codes.FailedPrecondition {
		t.Errorf("want %v, got %v (%v)", codes.FailedPrecondition, code, got)
	}
}
