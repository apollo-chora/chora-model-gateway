// Package secrets implements domain.SecretClient by reading credentials from
// the process environment.
//
// This is the whole credential mechanism: a registry entry names an
// environment variable (`api_key_env`) and the resolver reads it. No
// credential is ever compiled into the binary, written into the registry
// file, or baked into an image layer — which is what makes the same image
// safe to run against OpenAI, against a colleague's local vLLM, or against
// nothing at all.
package secrets

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
)

// EnvResolver resolves a credential reference to an environment variable and
// reads it.
type EnvResolver struct {
	// lookup is injectable so the resolver is testable without mutating the
	// real process environment.
	lookup func(string) (string, bool)

	mu    sync.RWMutex
	cache map[string]string
}

// NewEnvResolver constructs a resolver over the real process environment.
func NewEnvResolver() *EnvResolver {
	return &EnvResolver{
		lookup: os.LookupEnv,
		cache:  make(map[string]string),
	}
}

// NewResolverWithLookup constructs a resolver over an arbitrary lookup
// function. Used by tests.
func NewResolverWithLookup(lookup func(string) (string, bool)) *EnvResolver {
	return &EnvResolver{lookup: lookup, cache: make(map[string]string)}
}

// ResolveCredential implements domain.SecretClient.
//
// An unset reference returns ("", nil): the caller decides whether that is
// acceptable, and for a target with no `api_key_env` an empty credential is
// exactly right (vLLM, Ollama and LM Studio all serve without one). The
// service layer is what turns "configured but empty" into a refusal.
func (r *EnvResolver) ResolveCredential(_ context.Context, ref string) (string, error) {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return "", nil
	}

	r.mu.RLock()
	cached, ok := r.cache[ref]
	r.mu.RUnlock()
	if ok {
		return cached, nil
	}

	v, present := r.lookup(ref)
	if !present {
		return "", nil
	}
	v = strings.TrimSpace(v)

	r.mu.Lock()
	r.cache[ref] = v
	r.mu.Unlock()
	return v, nil
}

// InvalidateCache drops every memoised credential. Call it after a rotation
// so the next request picks the new value up without a restart.
func (r *EnvResolver) InvalidateCache() {
	r.mu.Lock()
	r.cache = make(map[string]string)
	r.mu.Unlock()
}

// Describe returns a non-sensitive description of what the resolver will do
// for a reference, for the boot log. It never returns the value.
func (r *EnvResolver) Describe(ref string) string {
	ref = strings.TrimSpace(ref)
	if ref == "" {
		return "no credential required"
	}
	v, present := r.lookup(ref)
	switch {
	case !present:
		return fmt.Sprintf("credential %s NOT SET", ref)
	case strings.TrimSpace(v) == "":
		return fmt.Sprintf("credential %s set but EMPTY", ref)
	default:
		return fmt.Sprintf("credential %s resolved", ref)
	}
}
