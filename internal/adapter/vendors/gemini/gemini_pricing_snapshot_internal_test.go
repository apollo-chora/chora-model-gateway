package gemini

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
)

// E7 shape (b) — dated-snapshot price resolution.
//
// WHY THIS EXISTS. E7 asked that FAMILIAR_MODEL "name an explicit version".
// Vertex publishes no dated snapshot of gemini-2.5-flash today (bare alias 200,
// -001 and -002 both 404), so nobody can satisfy it yet. The trap is what
// happens WHEN they can: geminiPricing is an exact map lookup, so the moment
// someone pins gemini-2.5-flash-002 the lookup misses, geminiCostMicros returns
// 0, and the Familiar bills $0 into the TokenUsageLedger AND debits $0 of tenant
// budget. Pinning would silently convert a priced model into a free one, which
// is worse than the drift E7 exists to prevent.
//
// This closes that trap ahead of the pin, so shape (a)'s reviewed pin can move
// to a real snapshot the day one ships without anyone remembering this file.
//
// WHAT IT MUST NOT DO. The table's own comments forbid guessing a rate, and two
// preview models are deliberately unpriced. Resolution therefore only ever maps
// an id onto an EXISTING curated key: it strips trailing NUMERIC segments and
// accepts the result only if it is already in the table. It never invents a
// price, never strips "-preview", and an explicit entry always wins.

func TestResolveGeminiPriceKey_ExactMatchWinsOverStripping(t *testing.T) {
	for _, model := range []string{"gemini-2.5-flash", "gemini-3-pro-image", "gemini-2.5-flash-lite"} {
		got, ok := resolveGeminiPriceKey(model)
		if !ok || got != model {
			t.Errorf("resolveGeminiPriceKey(%q) = (%q,%v), want (%q,true): a table key must resolve to itself", model, got, ok, model)
		}
	}
}

func TestResolveGeminiPriceKey_DatedSnapshotResolvesToItsBase(t *testing.T) {
	cases := map[string]string{
		"gemini-2.5-flash-001":      "gemini-2.5-flash",
		"gemini-2.5-flash-002":      "gemini-2.5-flash",
		"gemini-2.5-pro-001":        "gemini-2.5-pro",
		"gemini-3-pro-image-004":    "gemini-3-pro-image",
		"gemini-2.5-flash-lite-001": "gemini-2.5-flash-lite",
	}
	for model, want := range cases {
		got, ok := resolveGeminiPriceKey(model)
		if !ok || got != want {
			t.Errorf("resolveGeminiPriceKey(%q) = (%q,%v), want (%q,true)", model, got, ok, want)
		}
	}
}

func TestResolveGeminiPriceKey_NeverStripsPreview(t *testing.T) {
	// Both are deliberately unpriced. Stripping "-preview" would silently price
	// them off a base rate that has NOT been confirmed to apply, which is the
	// over-billing fabrication the table's comments forbid.
	for _, model := range []string{"gemini-3.1-pro-preview", "gemini-3-flash-preview"} {
		if got, ok := resolveGeminiPriceKey(model); ok {
			t.Errorf("resolveGeminiPriceKey(%q) resolved to %q — preview models must stay unpriced", model, got)
		}
	}
}

func TestResolveGeminiPriceKey_DatedPreviewStaysUnpriced(t *testing.T) {
	// gemini-2.5-flash-preview-05-20 strips its date segments to
	// "gemini-2.5-flash-preview", which is NOT a table key, so it must stop
	// there rather than continue stripping down to the priced base.
	if got, ok := resolveGeminiPriceKey("gemini-2.5-flash-preview-05-20"); ok {
		t.Errorf("dated PREVIEW resolved to %q — a preview must not inherit its base rate", got)
	}
}

func TestResolveGeminiPriceKey_UnknownFamilyDoesNotResolve(t *testing.T) {
	for _, model := range []string{"gemini-9-imaginary", "gemini-9-imaginary-001", "", "gemini", "gemini-3"} {
		if got, ok := resolveGeminiPriceKey(model); ok {
			t.Errorf("resolveGeminiPriceKey(%q) resolved to %q, want no resolution", model, got)
		}
	}
}

func TestResolveGeminiPriceKey_MalformedTrailingDashDoesNotResolve(t *testing.T) {
	// "gemini-2.5-flash-" yields an EMPTY trailing segment. An empty segment is
	// not digits, so it must stop rather than strip the dash and land on the
	// priced base: a malformed id is a caller bug and should surface as the
	// unpriced warn, not be silently repaired into a bill.
	if got, ok := resolveGeminiPriceKey("gemini-2.5-flash-"); ok {
		t.Errorf("resolveGeminiPriceKey(%q) resolved to %q, want no resolution", "gemini-2.5-flash-", got)
	}
}

func TestIsAllDigits(t *testing.T) {
	for _, s := range []string{"0", "001", "20", "1234"} {
		if !isAllDigits(s) {
			t.Errorf("isAllDigits(%q) = false, want true", s)
		}
	}
	for _, s := range []string{"", "preview", "05a", "a05", "2.5", "-1"} {
		if isAllDigits(s) {
			t.Errorf("isAllDigits(%q) = true, want false", s)
		}
	}
}

func TestResolveGeminiPriceKey_NeverStripsIntoADifferentFamily(t *testing.T) {
	// The guard that makes this safe: resolution only ever lands on a key that
	// is ALREADY in the curated table, so it cannot fabricate a rate for a model
	// family nobody priced.
	for model := range map[string]struct{}{"gemini-4-flash-001": {}, "gemini-5-pro-002": {}} {
		if got, ok := resolveGeminiPriceKey(model); ok {
			t.Errorf("resolveGeminiPriceKey(%q) resolved to %q, want no resolution for an unpriced family", model, got)
		}
	}
}

// ── the behaviour that actually matters: the snapshot bills like its base ────

func TestGeminiCostMicros_SnapshotPricesIdenticallyToItsBase(t *testing.T) {
	const oneMillion = 1_000_000
	base := geminiCostMicros("gemini-2.5-flash", oneMillion, oneMillion, 0)
	snap := geminiCostMicros("gemini-2.5-flash-002", oneMillion, oneMillion, 0)
	if base <= 0 {
		t.Fatalf("base model priced at %d micros — fixture is wrong, not the code", base)
	}
	if snap != base {
		t.Errorf("gemini-2.5-flash-002 = %d micros, gemini-2.5-flash = %d: a dated snapshot must bill as its base, "+
			"otherwise pinning a version silently makes the model free", snap, base)
	}
}

func TestGeminiCostMicros_SnapshotDoesNotWarnUnpriced(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	if got := geminiCostMicros("gemini-2.5-flash-002", 1_000, 500, 0); got <= 0 {
		t.Fatalf("resolved snapshot cost = %d micros, want > 0", got)
	}
	if line := strings.TrimSpace(buf.String()); line != "" {
		t.Errorf("a RESOLVED snapshot logged the unpriced warn, which would page on a correctly priced call: %s", line)
	}
}

func TestGeminiCostMicros_UnresolvableSnapshotStillWarnsUnpriced(t *testing.T) {
	// The regression guard for the whole change: adding resolution must not
	// swallow the CHO-2220 alert for a genuinely unknown model.
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	if got := geminiCostMicros("gemini-9-imaginary-001", 1_000, 500, 0); got != 0 {
		t.Fatalf("unresolvable model cost = %d micros, want 0", got)
	}
	line := strings.TrimSpace(buf.String())
	if line == "" {
		t.Fatal("unresolvable model produced NO warn — the $0 is silent again")
	}
	var rec map[string]any
	if err := json.Unmarshal([]byte(strings.Split(line, "\n")[0]), &rec); err != nil {
		t.Fatalf("warn line is not JSON: %v", err)
	}
	if flag, ok := rec["chora_unpriced_model"].(bool); !ok || !flag {
		t.Errorf("warn lost the stable chora_unpriced_model field the log-based metric keys on; got %v", rec["chora_unpriced_model"])
	}
	// The warn must name what the CALLER asked for, not a half-stripped stem —
	// otherwise the alert points at a model id that was never requested.
	if m, _ := rec["model"].(string); m != "gemini-9-imaginary-001" {
		t.Errorf("warn names model=%q, want the caller's id %q", m, "gemini-9-imaginary-001")
	}
}
