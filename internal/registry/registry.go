package registry

import (
	"fmt"
	"strings"
)

// Registry is the resolved, boot-validated model catalogue. Names (ids and
// aliases) resolve case-insensitively; the zero value of Get's second
// return means the name is not in the registry.
type Registry interface {
	// Get returns the spec registered under id (or one of its aliases),
	// matched case-insensitively. The bool reports whether the name exists.
	Get(id string) (ModelSpec, bool)

	// List returns every canonical spec in the registry, sorted by ID. Each
	// entry appears once regardless of how many aliases it declares.
	List() []ModelSpec

	// Resolve returns the primary spec for id plus the ordered fallback
	// chain: the caller's own chain first (deduplicated), then the
	// registry-declared FallbackIDs (deduplicated). An unknown name — the
	// primary, a caller fallback, or a registry fallback — fails the whole
	// resolve rather than being silently dropped.
	Resolve(id string, callerFallbacks []string) (primary ModelSpec, chain []ModelSpec, err error)
}

// memoryRegistry is the in-memory Registry implementation built by the
// loader. Read-only after construction; safe for concurrent use.
type memoryRegistry struct {
	byKey map[string]ModelSpec // lowercased id/alias → spec
	specs []ModelSpec          // canonical specs, sorted by ID
}

func (r *memoryRegistry) Get(id string) (ModelSpec, bool) {
	spec, ok := r.byKey[normalizeName(id)]
	return spec, ok
}

func (r *memoryRegistry) List() []ModelSpec {
	out := make([]ModelSpec, len(r.specs))
	copy(out, r.specs)
	return out
}

func (r *memoryRegistry) Resolve(id string, callerFallbacks []string) (ModelSpec, []ModelSpec, error) {
	primary, ok := r.Get(id)
	if !ok {
		return ModelSpec{}, nil, fmt.Errorf("policy: model %q is not in the registry", id)
	}
	chain := make([]ModelSpec, 0, len(callerFallbacks)+len(primary.FallbackIDs))
	seen := map[string]bool{primary.ID: true}
	for _, name := range callerFallbacks {
		candidate, ok := r.Get(name)
		if !ok {
			return ModelSpec{}, nil, fmt.Errorf("policy: model %q is not in the registry", name)
		}
		if seen[candidate.ID] {
			continue
		}
		seen[candidate.ID] = true
		chain = append(chain, candidate)
	}
	for _, name := range primary.FallbackIDs {
		candidate, ok := r.Get(name)
		if !ok {
			return ModelSpec{}, nil, fmt.Errorf("policy: model %q is not in the registry", name)
		}
		if seen[candidate.ID] {
			continue
		}
		seen[candidate.ID] = true
		chain = append(chain, candidate)
	}
	return primary, chain, nil
}

// normalizeName lowercases + trims a lookup key, matching the Python
// registry's case-insensitive name handling.
func normalizeName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}
