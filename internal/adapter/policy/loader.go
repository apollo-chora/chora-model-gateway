// Package policy adapts the model registry to domain.PolicyLoader.
//
// The loader is where a caller-supplied model NAME becomes a resolved
// upstream destination. It is deliberately thin: the registry already
// validated every entry at boot, so resolution here is a map lookup plus
// the fallback-chain expansion. An unknown model name is an error, never a
// silent substitution — a caller that asked for a cheap model must not be
// quietly given an expensive one.
package policy

import (
	"context"
	"fmt"

	"github.com/apollo-chora/chora-model-gateway/internal/config"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// Loader resolves InvokeRequests against a config.Registry.
type Loader struct {
	registry *config.Registry
}

// NewLoader constructs a Loader over a registry. The registry MUST NOT be
// nil — a loader with no catalogue would reject every call at runtime, which
// is a boot-time mistake.
func NewLoader(reg *config.Registry) (*Loader, error) {
	if reg == nil {
		return nil, fmt.Errorf("policy: model registry required")
	}
	return &Loader{registry: reg}, nil
}

// ResolveAgentPolicy implements domain.PolicyLoader.
//
// requestedModel must name a registry entry. fallbackModels, when supplied,
// are resolved through the registry exactly like the primary, so an agent
// may declare a chain that crosses providers. A fallback name that is not in
// the registry fails the whole resolve rather than being dropped: a silently
// shortened chain turns a degraded call into a hard failure at the worst
// possible moment.
func (l *Loader) ResolveAgentPolicy(
	_ context.Context,
	agentID, _ string,
	requestedModel domain.LogicalModelID,
	fallbackModels []domain.LogicalModelID,
) (domain.AgentPolicy, error) {
	primary, chain, err := l.resolve(requestedModel, fallbackModels)
	if err != nil {
		return domain.AgentPolicy{}, err
	}
	return domain.AgentPolicy{
		AgentID:                agentID,
		ResolvedLogicalModelID: primary.LogicalModelID,
		Target:                 primary,
		FallbackChain:          chain,
	}, nil
}

// resolve looks up one model and expands its declared fallback chain.
func (l *Loader) resolve(id domain.LogicalModelID, declaredFallbacks []domain.LogicalModelID) (domain.TargetModel, []domain.AgentPolicyFallback, error) {
	target, ok := l.registry.Lookup(id)
	if !ok {
		return domain.TargetModel{}, nil, fmt.Errorf("policy: model %q is not in the registry", id)
	}

	// Two sources of fallback chain: the caller's own declaration on the
	// request, and the registry entry's `fallback_ids`.
	//
	// The caller's chain comes FIRST. It is a per-dispatch decision — the
	// caller knows this turn is time-sensitive, or is running a batch where a
	// slow model is acceptable — and a per-dispatch decision should outrank a
	// deployment-wide default. Order IS the mechanism here: the first entry is
	// tried first.
	chain := make([]domain.AgentPolicyFallback, 0, 8)
	seen := map[domain.LogicalModelID]bool{target.LogicalModelID: true}
	appendTarget := func(fb domain.TargetModel) {
		if seen[fb.LogicalModelID] {
			return
		}
		seen[fb.LogicalModelID] = true
		chain = append(chain, domain.AgentPolicyFallback{
			Vendor:                 fb.Vendor,
			ResolvedLogicalModelID: fb.LogicalModelID,
			Target:                 fb,
		})
	}

	// Caller-declared, in the order given.
	for _, name := range declaredFallbacks {
		fb, ok := l.registry.Lookup(name)
		if !ok {
			// Fail the whole resolve rather than dropping the entry: a
			// silently shortened chain turns a degraded call into a hard
			// failure at the worst possible moment.
			return domain.TargetModel{}, nil, fmt.Errorf("policy: fallback model %q is not in the registry", name)
		}
		appendTarget(fb)
	}

	// Then the deployment default, for whatever the caller did not name.
	for _, fb := range l.registry.FallbacksFor(target.LogicalModelID) {
		appendTarget(fb)
	}
	return target, chain, nil
}
