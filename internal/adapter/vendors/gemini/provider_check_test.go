package gemini_test

import (
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/gemini"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// Compile-time checks that gemini.Provider implements the capability
// interfaces it still backs. GroundedGenerator/CapabilityProvider are NOT
// asserted here anymore — the grounded vendor port moved to the exa adapter
// (ADR-231 Exa cutover 2026-10-10); see internal/adapter/vendors/exa for
// its GroundedVendorClient check.
var _ domain.TextGenerator = (*gemini.Provider)(nil)
var _ domain.Embedder = (*gemini.Provider)(nil)
var _ domain.ImageGenerator = (*gemini.Provider)(nil)

func TestProvider_ImplementsDomainCapabilityProvider(t *testing.T) {
	// The compile-time checks above are the real verification.
	// This test exists so `go test` has something to run.
}
