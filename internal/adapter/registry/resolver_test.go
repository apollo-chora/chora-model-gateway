package registry

import (
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	modelregistry "github.com/apollo-chora/chora-model-gateway/internal/registry"
)

// The registry's provider token is the WIRE SHAPE ("openai"); the domain's
// VendorFamily is the ADAPTER FAMILY ("openai_byoa") that the dispatch tables
// are keyed by. Passing the token through verbatim left every
// registry-resolved model with a family no adapter is registered under, so the
// Embed flow could never find an adapter for a resolved provider.
func TestToModelInfo_MapsTheProviderTokenToTheDomainVendorFamily(t *testing.T) {
	cases := map[string]domain.VendorFamily{
		"openai":    domain.VendorFamilyOpenAI,
		"anthropic": domain.VendorFamilyAnthropic,
	}
	for provider, want := range cases {
		spec := modelregistry.ModelSpec{
			ID:            "m",
			Provider:      provider,
			UpstreamModel: "m",
			BaseURL:       "https://example.test/v1",
		}
		info := toModelInfo(spec)
		if info.Vendor != want {
			t.Errorf("provider %q: want domain family %q, got %q", provider, want, info.Vendor)
		}
	}
}

// The embedding dimension metadata reaches the domain unchanged: it is the
// only place the Embed flow can learn a model's vector length.
func TestToModelInfo_CarriesEmbeddingDimensionMetadata(t *testing.T) {
	spec := modelregistry.ModelSpec{
		ID:                       "embed",
		Provider:                 "openai",
		UpstreamModel:            "liquid/lfm-2.5-embedding-350m:free",
		BaseURL:                  "https://openrouter.ai/api/v1",
		Capabilities:             []string{"embeddings"},
		EmbeddingDimensions:      1024,
		EmbeddingDimensionsParam: false,
	}
	info := toModelInfo(spec)
	if info.EmbeddingDimensions != 1024 {
		t.Errorf("embedding_dimensions lost: got %d", info.EmbeddingDimensions)
	}
	if info.EmbeddingDimensionsParam {
		t.Error("a model that takes no dimensions override must report false")
	}
}
