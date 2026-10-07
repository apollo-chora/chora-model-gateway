// Package registry provides the model-registry adapter for the domain
// ModelResolver port. It wraps the in-memory registry (loaded from YAML +
// role-based env config) and exposes the dispatch metadata the Invoke
// flow needs: capabilities, output ceiling, credentials, grounding.
package registry

import (
	"context"
	"fmt"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
	modelregistry "github.com/5007-Capstone/chora/services/chora-model-gateway/internal/registry"
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
		ID:             spec.ID,
		Vendor:         domain.VendorFamily(spec.Provider),
		Capabilities:   spec.Capabilities,
		MaxOutputTokens: spec.MaxOutputTokens,
		APIKeyEnv:      spec.APIKeyEnv,
		APIKey:         spec.APIKey(),
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
