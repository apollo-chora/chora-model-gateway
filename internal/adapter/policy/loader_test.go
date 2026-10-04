package policy

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/config"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// registryYAML is the catalogue every test in this file resolves against. It
// carries one model per shape that matters: a primary with a declared
// registry-side chain, a cross-provider target a caller can name, a model with
// no chain at all, and a self-referential entry so the cycle guard has
// something to break.
const registryYAML = `
models:
  - id: primary
    provider: openai
    upstream_model: gpt-4o
    base_url: https://primary.test/v1
    api_key_env: PRIMARY_KEY
    context_window: 128000
    max_output_tokens: 4096
    capabilities: [chat, tools]
    fallback_ids: [registry-next]
    pricing:
      input_per_mtok_usd_micros: 150000
      output_per_mtok_usd_micros: 600000

  - id: registry-next
    provider: openai
    base_url: https://next.test/v1
    capabilities: [chat]

  - id: caller-next
    provider: anthropic
    base_url: https://caller.test
    messages_path: /v1/messages
    capabilities: [chat]

  - id: solo
    provider: openai
    base_url: https://solo.test/v1
    capabilities: [chat]

  - id: self-referential
    provider: openai
    base_url: https://loop.test/v1
    fallback_ids: [self-referential, solo]
    capabilities: [chat]
`

func newLoader(t *testing.T) *Loader {
	t.Helper()
	path := t.TempDir() + "/models.yaml"
	if err := os.WriteFile(path, []byte(registryYAML), 0o600); err != nil {
		t.Fatalf("write registry: %v", err)
	}
	reg, err := config.LoadRegistry(path)
	if err != nil {
		t.Fatalf("registry: %v", err)
	}
	l, err := NewLoader(reg)
	if err != nil {
		t.Fatalf("NewLoader: %v", err)
	}
	return l
}

func chainIDs(p domain.AgentPolicy) []string {
	out := make([]string, 0, len(p.FallbackChain))
	for _, fb := range p.FallbackChain {
		out = append(out, string(fb.ResolvedLogicalModelID))
	}
	return out
}

func TestNewLoader_RequiresARegistry(t *testing.T) {
	// A loader with no catalogue would reject every call at runtime. That is a
	// boot-time mistake, so it is refused at construction rather than turning
	// into a 100% failure rate after the service has already looked healthy.
	l, err := NewLoader(nil)
	if err == nil {
		t.Fatal("expected an error for a nil registry")
	}
	if l != nil {
		t.Errorf("loader = %v, want nil alongside the error", l)
	}
	if !strings.Contains(err.Error(), "model registry required") {
		t.Errorf("error = %q, want it to name the missing registry", err)
	}
}

func TestResolve_CarriesTheResolvedTarget(t *testing.T) {
	l := newLoader(t)

	p, err := l.ResolveAgentPolicy(context.Background(), "qgen_question", "content", "primary", nil)
	if err != nil {
		t.Fatalf("ResolveAgentPolicy: %v", err)
	}

	if p.AgentID != "qgen_question" {
		t.Errorf("agent id = %q, want it echoed back for the ledger row", p.AgentID)
	}
	if p.ResolvedLogicalModelID != "primary" {
		t.Errorf("resolved id = %q", p.ResolvedLogicalModelID)
	}
	// The Target is the whole point of the resolve: the service dispatches on
	// it without ever touching the registry itself, so every field the vendor
	// adapter needs must have survived.
	tgt := p.Target
	if tgt.Vendor != domain.VendorFamilyOpenAI {
		t.Errorf("vendor = %q", tgt.Vendor)
	}
	if tgt.UpstreamModel != "gpt-4o" {
		t.Errorf("upstream model = %q, want the alias target", tgt.UpstreamModel)
	}
	if tgt.BaseURL != "https://primary.test/v1" {
		t.Errorf("base url = %q", tgt.BaseURL)
	}
	if tgt.APIKeyRef != "PRIMARY_KEY" {
		t.Errorf("credential ref = %q, want the env var name, never the value", tgt.APIKeyRef)
	}
	if tgt.MaxOutputTokens != 4096 || tgt.ContextWindow != 128000 {
		t.Errorf("limits = %d / %d", tgt.MaxOutputTokens, tgt.ContextWindow)
	}
	if !tgt.Supports(domain.CapabilityTools) {
		t.Errorf("capabilities = %v, want the declared tools capability", tgt.Capabilities)
	}
}

func TestResolve_LookupIsCaseInsensitive(t *testing.T) {
	// Model names arrive in JSON bodies typed by hand, so "Primary" and
	// "PRIMARY" must resolve to the same entry rather than failing a call over
	// capitalisation.
	l := newLoader(t)

	for _, name := range []string{"primary", "Primary", "PRIMARY", " primary "} {
		p, err := l.ResolveAgentPolicy(context.Background(), "a", "", domain.LogicalModelID(name), nil)
		if err != nil {
			t.Fatalf("ResolveAgentPolicy(%q): %v", name, err)
		}
		if p.Target.UpstreamModel != "gpt-4o" {
			t.Errorf("ResolveAgentPolicy(%q) resolved to %+v", name, p.Target)
		}
	}
}

func TestResolve_UnknownModelFailsAndNamesIt(t *testing.T) {
	// A silent substitution is the failure mode this loader exists to prevent:
	// a caller who asked for a cheap model must never be quietly given an
	// expensive one, so the error has to name what they asked for.
	l := newLoader(t)

	p, err := l.ResolveAgentPolicy(context.Background(), "a", "", "no-such-model", nil)
	if err == nil {
		t.Fatal("expected an error for a model that is not in the registry")
	}
	if !strings.Contains(err.Error(), "no-such-model") {
		t.Errorf("error = %q, want it to name the unknown model", err)
	}
	if p.Target.UpstreamModel != "" || p.ResolvedLogicalModelID != "" {
		t.Errorf("a failed resolve returned a usable policy: %+v", p)
	}
}

func TestResolve_CallerDeclaredFallbacksAreHonoured(t *testing.T) {
	// A caller may name a chain the registry knows nothing about, and it may
	// cross providers: the anthropic target needs the Anthropic adapter, and a
	// resolve that dropped it would leave the fallback chain with nothing to
	// walk.
	l := newLoader(t)

	p, err := l.ResolveAgentPolicy(context.Background(), "a", "", "primary",
		[]domain.LogicalModelID{"caller-next"})
	if err != nil {
		t.Fatalf("ResolveAgentPolicy: %v", err)
	}

	got := chainIDs(p)
	if len(got) != 2 {
		t.Fatalf("chain = %v, want both the registry and the caller-declared fallback", got)
	}
	var callerFallback domain.AgentPolicyFallback
	for _, fb := range p.FallbackChain {
		if fb.ResolvedLogicalModelID == "caller-next" {
			callerFallback = fb
		}
	}
	if callerFallback.Target.UpstreamModel != "caller-next" {
		t.Errorf("caller fallback target = %+v, want it fully resolved", callerFallback.Target)
	}
	if callerFallback.Vendor != domain.VendorFamilyAnthropic {
		t.Errorf("caller fallback vendor = %q, want anthropic", callerFallback.Vendor)
	}
	// The resolved target is what actually gets called, so the URL must be the
	// Anthropic-shaped one.
	if got := callerFallback.Target.MessagesURL(); got != "https://caller.test/v1/messages" {
		t.Errorf("fallback messages url = %q", got)
	}
}

// TestResolve_CallersChainIsOrderedBeforeTheRegistrysChain pins the ORDER of
// the fallback chain, because order is the entire mechanism: the first entry is
// the first thing tried.
//
// The caller's chain leads. A caller that names its own fallback is making a
// per-dispatch decision, and that outranks the deployment-wide default an
// operator wrote into the registry file.
func TestResolve_CallersChainIsOrderedBeforeTheRegistrysChain(t *testing.T) {
	l := newLoader(t)

	p, err := l.ResolveAgentPolicy(context.Background(), "a", "", "primary",
		[]domain.LogicalModelID{"caller-next"})
	if err != nil {
		t.Fatalf("ResolveAgentPolicy: %v", err)
	}

	got := chainIDs(p)
	want := []string{"caller-next", "registry-next"}
	if len(got) != len(want) {
		t.Fatalf("chain = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("chain[%d] = %q, want %q (order: %v)", i, got[i], want[i], got)
		}
	}
}

func TestResolve_UnknownCallerFallbackFailsTheWholeResolve(t *testing.T) {
	// A silently shortened chain turns a degraded call into a hard failure at
	// the worst possible moment — the caller only discovers the gap when the
	// primary is already down. So a bad name in the chain is an error now.
	l := newLoader(t)

	p, err := l.ResolveAgentPolicy(context.Background(), "a", "", "primary",
		[]domain.LogicalModelID{"caller-next", "typo-model"})
	if err == nil {
		t.Fatal("expected an error when one declared fallback is unknown")
	}
	if !strings.Contains(err.Error(), "typo-model") {
		t.Errorf("error = %q, want it to name the unknown fallback", err)
	}
	// Nothing is handed back: a partially-populated chain would look like a
	// valid degraded route to the caller.
	if len(p.FallbackChain) != 0 || p.Target.UpstreamModel != "" {
		t.Errorf("a failed resolve returned a partial policy: %+v", p)
	}
}

func TestResolve_UnknownPrimaryIsNotReportedAsAFallbackError(t *testing.T) {
	// The two failures are different: an unknown primary is a caller bug, an
	// unknown fallback is a chain-definition bug. The message says which.
	l := newLoader(t)

	_, err := l.ResolveAgentPolicy(context.Background(), "a", "", "no-such-model",
		[]domain.LogicalModelID{"also-missing"})
	if err == nil {
		t.Fatal("expected an error")
	}
	if !strings.Contains(err.Error(), "model") || !strings.Contains(err.Error(), "no-such-model") {
		t.Errorf("error = %q, want it to name the primary model", err)
	}
}

func TestResolve_DuplicateAndCyclicFallbacksAreSkipped(t *testing.T) {
	// A chain that repeats a model would re-dispatch to a target already known
	// to be failing, and a self-referential registry entry would otherwise
	// recurse. Both are silently dropped rather than refused: a redundant name
	// is harmless, and refusing it would break working configurations.
	l := newLoader(t)

	p, err := l.ResolveAgentPolicy(context.Background(), "a", "", "self-referential",
		[]domain.LogicalModelID{"self-referential", "solo", "solo"})
	if err != nil {
		t.Fatalf("ResolveAgentPolicy: %v", err)
	}
	got := chainIDs(p)
	if len(got) != 1 || got[0] != "solo" {
		t.Errorf("chain = %v, want the self-reference and the duplicate dropped", got)
	}
}

func TestResolve_CallerCannotReDispatchToThePrimary(t *testing.T) {
	// The primary is already marked seen, so naming it as its own fallback adds
	// nothing but a wasted attempt.
	l := newLoader(t)

	p, err := l.ResolveAgentPolicy(context.Background(), "a", "", "primary",
		[]domain.LogicalModelID{"primary"})
	if err != nil {
		t.Fatalf("ResolveAgentPolicy: %v", err)
	}
	if got := chainIDs(p); len(got) != 1 || got[0] != "registry-next" {
		t.Errorf("chain = %v, want the primary not repeated", got)
	}
}

func TestResolve_NoFallbacksIsSingleShot(t *testing.T) {
	// An empty chain means the service will not walk anything on a vendor
	// failure; an entry with no fallback_ids and no declared fallbacks is the
	// normal configuration, not an error.
	l := newLoader(t)

	p, err := l.ResolveAgentPolicy(context.Background(), "a", "", "solo", nil)
	if err != nil {
		t.Fatalf("ResolveAgentPolicy: %v", err)
	}
	if len(p.FallbackChain) != 0 {
		t.Errorf("chain = %v, want empty", chainIDs(p))
	}
	if p.Target.UpstreamModel != "solo" {
		t.Errorf("target = %+v", p.Target)
	}
}

func TestResolve_EmptyChainMatchesNoChain(t *testing.T) {
	// An agent that declares an explicitly empty slice is in the same position
	// as one that declares nothing.
	l := newLoader(t)

	p, err := l.ResolveAgentPolicy(context.Background(), "a", "", "solo",
		[]domain.LogicalModelID{})
	if err != nil {
		t.Fatalf("ResolveAgentPolicy: %v", err)
	}
	if len(p.FallbackChain) != 0 {
		t.Errorf("chain = %v, want empty", chainIDs(p))
	}
}

func TestResolve_RepeatedResolvesAreDeterministic(t *testing.T) {
	// The registry never changes at runtime, so two resolves of the same request
	// must produce the same chain — the fallback walk is only predictable if the
	// order does not shift between attempts of the same call.
	l := newLoader(t)

	first, err := l.ResolveAgentPolicy(context.Background(), "a", "", "primary",
		[]domain.LogicalModelID{"caller-next", "solo"})
	if err != nil {
		t.Fatalf("first resolve: %v", err)
	}
	for i := 0; i < 5; i++ {
		again, err := l.ResolveAgentPolicy(context.Background(), "a", "", "primary",
			[]domain.LogicalModelID{"caller-next", "solo"})
		if err != nil {
			t.Fatalf("repeat %d: %v", i, err)
		}
		if strings.Join(chainIDs(again), ",") != strings.Join(chainIDs(first), ",") {
			t.Fatalf("chain changed between resolves: %v vs %v", chainIDs(again), chainIDs(first))
		}
	}
}

func TestLoaderSatisfiesThePolicyLoaderPort(t *testing.T) {
	// The service layer depends on the port, not the concrete type; a wiring
	// mistake here would only surface at the first dispatch.
	var _ domain.PolicyLoader = newLoader(t)
}
