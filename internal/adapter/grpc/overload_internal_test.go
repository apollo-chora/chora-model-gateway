package modelgatewaygrpc

import (
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// The per-tenant concurrency ceiling (CHORA_LLM_MAX_CONCURRENT_PER_TENANT) must
// surface as RESOURCE_EXHAUSTED on all three dispatch paths: the request never
// reached a provider, so it is retryable, not a vendor failure.
func TestMapOverloadedError_ResourceExhaustedOnEveryPath(t *testing.T) {
	err := &domain.OverloadedError{TenantID: "tenant-1", Limit: 1}
	cases := map[string]error{
		"invoke":   mapInvokeError(err),
		"embed":    mapEmbedError(err),
		"grounded": mapGroundedError(err),
	}
	for name, got := range cases {
		if code := status.Code(got); code != codes.ResourceExhausted {
			t.Errorf("%s: want %v, got %v (%v)", name, codes.ResourceExhausted, code, got)
		}
	}
}

// A GroundedSearchError that wraps the overload (the shape the domain service
// returns) maps the same way.
func TestMapGroundedError_OverloadWrappedInGroundedSearchError(t *testing.T) {
	got := mapGroundedError(&domain.GroundedSearchError{
		Reason: domain.EgressDenyOverloaded,
		Detail: "limit reached",
		Inner:  &domain.OverloadedError{TenantID: "tenant-1", Limit: 1},
	})
	if code := status.Code(got); code != codes.ResourceExhausted {
		t.Errorf("want %v, got %v", codes.ResourceExhausted, code)
	}
}
