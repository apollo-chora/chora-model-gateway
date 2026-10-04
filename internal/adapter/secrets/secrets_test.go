package secrets

import (
	"context"
	"strings"
	"sync"
	"testing"
)

// countingLookup wraps a map as a lookup function and counts the calls, so a
// test can prove memoisation actually happened rather than inferring it from a
// correct value.
type countingLookup struct {
	mu    sync.Mutex
	vals  map[string]string
	calls map[string]int
}

func newCountingLookup(vals map[string]string) *countingLookup {
	return &countingLookup{vals: vals, calls: map[string]int{}}
}

func (c *countingLookup) lookup(ref string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls[ref]++
	v, ok := c.vals[ref]
	return v, ok
}

func (c *countingLookup) timesCalled(ref string) int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls[ref]
}

func (c *countingLookup) totalCalls() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	n := 0
	for _, v := range c.calls {
		n += v
	}
	return n
}

func TestResolveCredential_UnsetReferenceIsEmptyNotAnError(t *testing.T) {
	// An unset reference is not necessarily a fault: a local vLLM / Ollama /
	// LM Studio server needs no key at all, and the target with no api_key_env
	// must resolve to exactly this. The service layer, which knows whether the
	// target was CONFIGURED to have a key, is what turns "configured but empty"
	// into a refusal.
	r := NewResolverWithLookup(newCountingLookup(nil).lookup)

	got, err := r.ResolveCredential(context.Background(), "ANTHROPIC_API_KEY")
	if err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	if got != "" {
		t.Errorf("credential = %q, want empty", got)
	}
}

func TestResolveCredential_EmptyReferenceNeverTouchesTheEnvironment(t *testing.T) {
	// A target with no api_key_env must not cost a lookup at all; a hot path
	// that stats the environment for a ref that cannot exist is pure waste.
	cl := newCountingLookup(nil)
	r := NewResolverWithLookup(cl.lookup)

	got, err := r.ResolveCredential(context.Background(), "")
	if err != nil || got != "" {
		t.Fatalf("ResolveCredential(\"\") = %q, %v", got, err)
	}
	if cl.totalCalls() != 0 {
		t.Errorf("lookup was called %d times for an empty reference", cl.totalCalls())
	}
}

func TestResolveCredential_WhitespaceReferenceIsTreatedAsEmpty(t *testing.T) {
	// A registry value that picked up a trailing newline must not become a
	// lookup for a variable named " KEY" that does not exist.
	cl := newCountingLookup(nil)
	r := NewResolverWithLookup(cl.lookup)

	if _, err := r.ResolveCredential(context.Background(), "  \t "); err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	if cl.totalCalls() != 0 {
		t.Errorf("lookup was called %d times for a whitespace-only reference", cl.totalCalls())
	}
}

func TestResolveCredential_ReturnsTheTrimmedValue(t *testing.T) {
	// Secrets pasted from a console or a .env file carry trailing newlines, and
	// a key with a stray "\n" is rejected by every provider.
	r := NewResolverWithLookup(newCountingLookup(map[string]string{
		"KEY": "  sk-ant-abc123  \n",
	}).lookup)

	got, err := r.ResolveCredential(context.Background(), "KEY")
	if err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	if got != "sk-ant-abc123" {
		t.Errorf("credential = %q, want it trimmed", got)
	}
}

func TestResolveCredential_WhitespaceOnlyValueReadsAsEmpty(t *testing.T) {
	// A variable that exists but holds only whitespace is indistinguishable
	// from an unset one as far as the provider is concerned — it must read as
	// empty so the service's "configured but empty" refusal fires.
	r := NewResolverWithLookup(newCountingLookup(map[string]string{
		"KEY": "   \n\t ",
	}).lookup)

	got, err := r.ResolveCredential(context.Background(), "KEY")
	if err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	if got != "" {
		t.Errorf("credential = %q, want empty for a whitespace-only value", got)
	}
}

func TestResolveCredential_ReferenceIsTrimmedBeforeLookup(t *testing.T) {
	cl := newCountingLookup(map[string]string{"KEY": "sk-x"})
	r := NewResolverWithLookup(cl.lookup)

	if _, err := r.ResolveCredential(context.Background(), "  KEY  "); err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	if cl.timesCalled("KEY") != 1 {
		t.Errorf("the trimmed name was not the one looked up: calls = %v", cl.calls)
	}
}

func TestResolveCredential_ReadsTheEnvironmentAtMostOncePerRef(t *testing.T) {
	// Every dispatch on a hot agent re-resolves the credential; re-reading the
	// environment per call turns a credential lookup into a syscall on the
	// request path. Memoisation is what makes the cache map worth having.
	cl := newCountingLookup(map[string]string{
		"A_KEY": "sk-a",
		"B_KEY": "sk-b",
	})
	r := NewResolverWithLookup(cl.lookup)

	for i := 0; i < 25; i++ {
		if _, err := r.ResolveCredential(context.Background(), "A_KEY"); err != nil {
			t.Fatalf("resolve %d: %v", i, err)
		}
		if _, err := r.ResolveCredential(context.Background(), "B_KEY"); err != nil {
			t.Fatalf("resolve %d: %v", i, err)
		}
	}
	if got := cl.timesCalled("A_KEY"); got != 1 {
		t.Errorf("A_KEY was read %d times, want 1", got)
	}
	if got := cl.timesCalled("B_KEY"); got != 1 {
		t.Errorf("B_KEY was read %d times, want 1", got)
	}
}

func TestResolveCredential_CachesPerRef(t *testing.T) {
	// Each reference is memoised independently, so two registry entries holding
	// different keys do not collide in the cache.
	cl := newCountingLookup(map[string]string{
		"A_KEY": "sk-a",
		"B_KEY": "sk-b",
	})
	r := NewResolverWithLookup(cl.lookup)

	for i := 0; i < 3; i++ {
		got, err := r.ResolveCredential(context.Background(), "A_KEY")
		if err != nil || got != "sk-a" {
			t.Fatalf("A_KEY = %q, %v", got, err)
		}
		got, err = r.ResolveCredential(context.Background(), "B_KEY")
		if err != nil || got != "sk-b" {
			t.Fatalf("B_KEY = %q, %v", got, err)
		}
	}
	if cl.totalCalls() != 2 {
		t.Errorf("total lookups = %d, want exactly one per distinct ref", cl.totalCalls())
	}
}

func TestResolveCredential_AnUnsetRefIsReRead(t *testing.T) {
	// A miss is NOT memoised, so a key injected after boot is picked up without
	// a restart. The cost is one environment read per dispatch for a ref that
	// does not exist yet — bounded, and worth paying for the self-healing.
	vals := map[string]string{}
	cl := newCountingLookup(vals)
	r := NewResolverWithLookup(cl.lookup)

	if _, err := r.ResolveCredential(context.Background(), "LATE_KEY"); err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	// The variable appears.
	vals["LATE_KEY"] = "sk-late"
	got, err := r.ResolveCredential(context.Background(), "LATE_KEY")
	if err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	if got != "sk-late" {
		t.Errorf("credential = %q, want the newly-set value without a restart", got)
	}
}

func TestInvalidateCache_ForcesAReRead(t *testing.T) {
	// Rotation is the whole reason the cache exists rather than a bare read:
	// after a rotation the next dispatch must see the new value, with no
	// restart and no manual cache surgery.
	vals := map[string]string{"KEY": "old-value"}
	cl := newCountingLookup(vals)
	r := NewResolverWithLookup(cl.lookup)

	if got, _ := r.ResolveCredential(context.Background(), "KEY"); got != "old-value" {
		t.Fatalf("first resolve = %q", got)
	}
	if got := cl.timesCalled("KEY"); got != 1 {
		t.Fatalf("reads before invalidation = %d, want 1", got)
	}

	vals["KEY"] = "new-value"
	if got, _ := r.ResolveCredential(context.Background(), "KEY"); got != "old-value" {
		t.Errorf("resolve before invalidation = %q, want the memoised value", got)
	}

	r.InvalidateCache()

	got, err := r.ResolveCredential(context.Background(), "KEY")
	if err != nil {
		t.Fatalf("ResolveCredential after invalidation: %v", err)
	}
	if got != "new-value" {
		t.Errorf("credential after invalidation = %q, want the rotated value", got)
	}
	if reads := cl.timesCalled("KEY"); reads != 2 {
		t.Errorf("KEY was read %d times, want 2 (one per generation)", reads)
	}
}

func TestInvalidateCache_OnAFreshResolverIsSafe(t *testing.T) {
	// Rotation happens on a timer; invalidating before anything has resolved
	// must not panic.
	r := NewResolverWithLookup(newCountingLookup(nil).lookup)
	r.InvalidateCache()
	r.InvalidateCache()

	if _, err := r.ResolveCredential(context.Background(), "ANY"); err != nil {
		t.Fatalf("ResolveCredential after a no-op invalidation: %v", err)
	}
}

func TestInvalidateCache_ClearsEveryRef(t *testing.T) {
	vals := map[string]string{"A_KEY": "sk-a1", "B_KEY": "sk-b1"}
	cl := newCountingLookup(vals)
	r := NewResolverWithLookup(cl.lookup)

	for i := 0; i < 3; i++ {
		_, _ = r.ResolveCredential(context.Background(), "A_KEY")
		_, _ = r.ResolveCredential(context.Background(), "B_KEY")
	}
	vals["A_KEY"] = "sk-a2"
	vals["B_KEY"] = "sk-b2"
	r.InvalidateCache()

	for ref, want := range map[string]string{"A_KEY": "sk-a2", "B_KEY": "sk-b2"} {
		got, err := r.ResolveCredential(context.Background(), ref)
		if err != nil {
			t.Fatalf("%s: %v", ref, err)
		}
		if got != want {
			t.Errorf("%s = %q, want %q", ref, got, want)
		}
	}
}

func TestDescribe_NeverLeaksTheValue(t *testing.T) {
	// Describe feeds the boot log, which is read by anyone with log access. A
	// description is only useful if it can be shown there safely, so it states
	// the state of the ref and nothing more.
	const secret = "sk-ant-super-secret-value"
	r := NewResolverWithLookup(newCountingLookup(map[string]string{
		"SET_KEY":    secret,
		"EMPTY_KEY":  "  ",
		"NOTSET_KEY": "",
	}).lookup)

	for _, ref := range []string{"", "   ", "SET_KEY", "EMPTY_KEY", "NOTSET_KEY"} {
		got := r.Describe(ref)
		if strings.Contains(got, secret) {
			t.Errorf("Describe(%q) leaked the credential: %q", ref, got)
		}
		if strings.Contains(got, "  ") && strings.TrimSpace(got) == "" {
			t.Errorf("Describe(%q) = %q, want a description", ref, got)
		}
	}
}

func TestDescribe_StatesEachCase(t *testing.T) {
	r := NewResolverWithLookup(newCountingLookup(map[string]string{
		"SET_KEY":   "sk-ant-value",
		"EMPTY_KEY": "   ",
	}).lookup)

	cases := []struct {
		ref      string
		wantSubs string
	}{
		// A target with no api_key_env at all: not a gap, just a local server.
		{"", "no credential required"},
		{"   ", "no credential required"},
		// Configured but the variable does not exist: an operator mistake.
		{"MISSING_KEY", "NOT SET"},
		// Configured, variable exists, but holds nothing usable.
		{"EMPTY_KEY", "EMPTY"},
		// The healthy case.
		{"SET_KEY", "resolved"},
	}
	for _, tc := range cases {
		got := r.Describe(tc.ref)
		if !strings.Contains(got, tc.wantSubs) {
			t.Errorf("Describe(%q) = %q, want it to contain %q", tc.ref, got, tc.wantSubs)
		}
	}
}

func TestDescribe_DoesNotWarmTheCache(t *testing.T) {
	// Describe is a diagnostic. If it memoised, the boot log's answer would
	// silently become the value served for the rest of the process's life,
	// including the "NOT SET" case where nothing was there to cache.
	cl := newCountingLookup(map[string]string{"KEY": "sk-x"})
	r := NewResolverWithLookup(cl.lookup)

	if got := r.Describe("KEY"); !strings.Contains(got, "resolved") {
		t.Fatalf("Describe = %q", got)
	}
	if _, err := r.ResolveCredential(context.Background(), "KEY"); err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	if got := cl.timesCalled("KEY"); got != 2 {
		t.Errorf("KEY was read %d times, want 2 (Describe must not warm the cache)", got)
	}
}

func TestNewEnvResolver_ReadsTheProcessEnvironment(t *testing.T) {
	// The production constructor is a thin wrapper over os.LookupEnv; it has to
	// work without any test-only wiring, so this goes through the real
	// environment via t.Setenv.
	t.Setenv("CHORA_TEST_SECRET", "  from-the-process-env  ")

	r := NewEnvResolver()
	got, err := r.ResolveCredential(context.Background(), "CHORA_TEST_SECRET")
	if err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	if got != "from-the-process-env" {
		t.Errorf("credential = %q, want the process value, trimmed", got)
	}

	// Rotation works through the same entrypoint.
	t.Setenv("CHORA_TEST_SECRET", "rotated")
	if got, _ := r.ResolveCredential(context.Background(), "CHORA_TEST_SECRET"); got != "from-the-process-env" {
		t.Errorf("before invalidation = %q, want the memoised value", got)
	}
	r.InvalidateCache()
	if got, _ := r.ResolveCredential(context.Background(), "CHORA_TEST_SECRET"); got != "rotated" {
		t.Errorf("after invalidation = %q, want the rotated value", got)
	}

	if got := r.Describe("CHORA_TEST_SECRET"); !strings.Contains(got, "resolved") {
		t.Errorf("Describe = %q", got)
	}
}

func TestNewEnvResolver_UnsetVariable(t *testing.T) {
	// os.Setenv with an empty value is still "present"; the resolver reads it as
	// an empty credential, which the service refuses. An entirely absent
	// variable is the case exercised here.
	r := NewEnvResolver()
	got, err := r.ResolveCredential(context.Background(), "CHORA_TEST_DEFINITELY_UNSET_VAR")
	if err != nil {
		t.Fatalf("ResolveCredential: %v", err)
	}
	if got != "" {
		t.Errorf("credential = %q, want empty", got)
	}
}

func TestResolveCredential_IsConcurrencySafe(t *testing.T) {
	// Dispatch is concurrent and the cache is shared, so the memoisation must
	// not race with itself. Run under -race this is the assertion that matters.
	cl := newCountingLookup(map[string]string{"KEY": "sk-x"})
	r := NewResolverWithLookup(cl.lookup)

	var wg sync.WaitGroup
	for i := 0; i < 16; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 20; j++ {
				if got, err := r.ResolveCredential(context.Background(), "KEY"); err != nil || got != "sk-x" {
					t.Errorf("resolve = %q, %v", got, err)
					return
				}
				r.Describe("KEY")
				if j%5 == 0 {
					r.InvalidateCache()
				}
			}
		}()
	}
	wg.Wait()
}
