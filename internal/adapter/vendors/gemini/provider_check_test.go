package gemini_test

import (
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/gemini"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// Compile-time checks that gemini.Provider implements domain.CapabilityProvider
// and each capability interface.
var _ domain.CapabilityProvider = (*gemini.Provider)(nil)
var _ domain.TextGenerator = (*gemini.Provider)(nil)
var _ domain.Embedder = (*gemini.Provider)(nil)
var _ domain.ImageGenerator = (*gemini.Provider)(nil)
var _ domain.GroundedGenerator = (*gemini.Provider)(nil)

func TestProvider_ImplementsDomainCapabilityProvider(t *testing.T) {
	// The compile-time checks above are the real verification.
	// This test exists so `go test` has something to run.
}
