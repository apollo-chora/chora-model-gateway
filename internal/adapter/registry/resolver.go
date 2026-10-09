// Package registry provides the model-registry adapter for the domain
// ModelResolver port. It wraps the in-memory registry (loaded from YAML +
// role-based env config) and exposes the dispatch metadata the Invoke
// flow needs: capabilities, output ceiling, credentials, grounding.
package registry

import (
	"context"
	"fmt"
	"strings"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	modelregistry "github.com/apollo-chora/chora-model-gateway/internal/registry"
)

// Resolver adapts a modelregistry.Registry to the domain.ModelResolver port.
// It is the production wiring: the domain service resolves model metadata
// through this adapter, keeping the domain decoupled from the registry
// implementation.
type Resolver struct {
	registry modelregistry.Registry
}

// NewResolver constructs a Resolver. The registry MUST be non-nil
// (fail-loud per feedback_no_stubs_real_wiring).
func NewResolver(r modelregistry.Registry) (*Resolver, error) {
	if r == nil {
		return nil, fmt.Errorf("registry: Registry required")
	}
	return &Resolver{registry: r}, nil
}

// Resolve implements domain.ModelResolver.
func (r *Resolver) Resolve(ctx context.Context, id domain.LogicalModelID) (domain.ModelInfo, error) {
	spec, ok := r.registry.Get(string(id))
	if !ok {
		return domain.ModelInfo{}, fmt.Errorf("policy: model %q is not in the registry", id)
	}
	return toModelInfo(spec), nil
}

// toModelInfo converts a modelregistry.ModelSpec to a domain.ModelInfo.
func toModelInfo(spec modelregistry.ModelSpec) domain.ModelInfo {
	info := domain.ModelInfo{
		ID:                       spec.ID,
		Vendor:                   vendorFamilyForProvider(spec.Provider),
		UpstreamModel:            spec.UpstreamModel,
		BaseURL:                  spec.BaseURL,
		Capabilities:             spec.Capabilities,
		EmbeddingDimensions:      spec.EmbeddingDimensions,
		EmbeddingDimensionsParam: spec.EmbeddingDimensionsParam,
		MaxOutputTokens:          spec.MaxOutputTokens,
		APIKeyEnv:                spec.APIKeyEnv,
		APIKey:                   spec.APIKey(),
	}
	if spec.Grounding != nil {
		info.Grounding = &domain.GroundingInfo{
			Surface:       spec.Grounding.Surface,
			ResponsesPath: spec.Grounding.ResponsesPath,
			ToolType:      spec.Grounding.ToolType,
			ToolName:      spec.Grounding.ToolName,
			MaxUses:       spec.Grounding.MaxUses,
		}
	}
	for _, fb := range spec.FallbackIDs {
		info.FallbackIDs = append(info.FallbackIDs, domain.LogicalModelID(fb))
	}
	return info
}

// vendorFamilyForProvider maps a registry provider token to the domain
// VendorFamily the dispatch tables are keyed by. The two vocabularies are NOT
// the same string: the registry names the WIRE SHAPE the adapter speaks
// ("openai", "anthropic"), while the domain names the ADAPTER FAMILY
// ("openai_byoa", "anthropic_byoa"). Passing the token through verbatim left
// every registry-resolved model with a family no adapter is registered under,
// so the Embed flow's per-family adapter table could never match a resolved
// provider. An unknown token is passed through lowercased, so a future
// provider fails loudly at dispatch instead of silently mapping somewhere.
func vendorFamilyForProvider(provider string) domain.VendorFamily {
	token := strings.ToLower(strings.TrimSpace(provider))
	switch token {
	case "openai":
		return domain.VendorFamilyOpenAI
	case "anthropic":
		return domain.VendorFamilyAnthropic
	default:
		return domain.VendorFamily(token)
	}
}
