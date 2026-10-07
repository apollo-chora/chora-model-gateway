package gemini

import (
	"bytes"
	"encoding/json"
	"log/slog"
	"strings"
	"testing"
)

// CHO-2220 — the gateway's Gemini price table is the ONLY thing that decides
// TokenUsageLedger cost_micros AND the per-tenant budget DebitSpent. Until this
// story it had no tests at all, which is how it drifted to 2.5-era rates that
// under-bill every model it covers and silently return 0 for every model it
// does not.
//
// AUTHORITY = the published Vertex AI price list
// (https://cloud.google.com/vertex-ai/generative-ai/pricing), cross-checked
// against the Billing Catalog API (service C7E2-9256-1C43 "Vertex AI"). Not
// from memory.
//
// ⚠ Where the two disagree, the PAGE wins — the SKU catalogue retains
// preview-era SKUs alongside GA ones and its names are ambiguous:
//
//   - "Gemini 2.5 Flash GA Text Input" ($0.30) vs "Gemini 2.5 Flash Text Input"
//     ($0.15). The un-suffixed one is the pre-GA price. gemini-2.5-flash is GA.
//   - "Gemini 2.5 Flash Image Input" ($0.15) reads as either (2.5 Flash Image)
//     (Input) or (2.5 Flash)(Image Input). It is the stale preview-era rate
//     either way — the page prices gemini-2.5-flash-image input at $0.30. This
//     cost us a 2x under-bill on the first pass of CHO-2220.
//   - Contrast "Gemini 3.0 Pro Image Input Caching" which sits beside
//     Text/Audio/Video Input Caching — there the suffix is the input MODALITY,
//     not the model. Same words, different grammar. Read the family, not the name.

// oneMillion is the token count that makes a per-1M rate assert as itself:
// cost(1M input tokens) == InputMicrosPer1M.
const oneMillion = int64(1_000_000)

// wantRate is one model's verified rates in micro-USD per 1M tokens.
type wantRate struct {
	inMicrosPer1M    int64
	outMicrosPer1M   int64
	cacheMicrosPer1M int64
	src              string // where each number came from
}

// vertexRates is the authority this package is tested against.
var vertexRates = map[string]wantRate{
	"gemini-2.5-flash": {
		300_000, 2_500_000, 30_000,
		"page: in $0.30 / out $2.50 / cached $0.03 per 1M",
	},
	"gemini-2.5-pro": {
		1_250_000, 10_000_000, 130_000,
		"page: in $1.25 / out $10.00 / cached $0.13 per 1M (<=200K context tier)",
	},
	"gemini-2.5-flash-lite": {
		100_000, 400_000, 10_000,
		"page: in $0.10 / out $0.40 / cached $0.01 per 1M",
	},
	"gemini-3.5-flash": {
		1_500_000, 9_000_000, 150_000,
		"page+SKU: in $1.50 / out $9.00 / cached $0.15 per 1M (Global; non-global is +10%)",
	},
	"gemini-3.1-flash-lite": {
		250_000, 1_500_000, 25_000,
		"SKU: in $0.25 / out $1.50 / cached $0.025 per 1M (Global)",
	},
	"gemini-2.5-flash-image": {
		300_000, 30_000_000, 30_000,
		"page: in $0.30 / image-out $30.00 per 1M. Input was WRONG at $0.15 on the " +
			"first CHO-2220 pass (stale preview-era SKU). cached $0.03 inferred from " +
			"the 2.5-flash base — no dedicated SKU or page entry; never exercised " +
			"(scene generation is single-turn, no context caching)",
	},
	"gemini-3.1-flash-image": {
		500_000, 60_000_000, 50_000,
		"page+SKU: in $0.50 / image-out $60.00 / cached $0.05 per 1M",
	},
	"gemini-3-pro-image": {
		2_000_000, 120_000_000, 200_000,
		"page+SKU: in $2.00 / image-out $120.00 / cached $0.20 per 1M. NB the page " +
			"prices TEXT output at $12/1M; we bill the image-output rate because the " +
			"qgen scene path always requests response_modality=IMAGE",
	},
}

// TestGeminiCostMicros_RatesMatchVertexSKUCatalogue pins every priced model to
// its live SKU rate. A rate that drifts from Google's catalogue fails here
// rather than silently mis-billing.
func TestGeminiCostMicros_RatesMatchVertexSKUCatalogue(t *testing.T) {
	for model, want := range vertexRates {
		t.Run(model+"/input", func(t *testing.T) {
			got := geminiCostMicros(model, oneMillion, 0, 0)
			if got != want.inMicrosPer1M {
				t.Errorf("input cost for 1M tokens = %d micros, want %d micros\n  src: %s",
					got, want.inMicrosPer1M, want.src)
			}
		})
		t.Run(model+"/output", func(t *testing.T) {
			got := geminiCostMicros(model, 0, oneMillion, 0)
			if got != want.outMicrosPer1M {
				t.Errorf("output cost for 1M tokens = %d micros, want %d micros\n  src: %s",
					got, want.outMicrosPer1M, want.src)
			}
		})
		t.Run(model+"/cached", func(t *testing.T) {
			// 1M prompt tokens, ALL cached ⇒ zero fresh input, so the whole
			// cost is the cache rate.
			got := geminiCostMicros(model, oneMillion, 0, oneMillion)
			if got != want.cacheMicrosPer1M {
				t.Errorf("fully-cached prompt cost for 1M tokens = %d micros, want %d micros\n  src: %s",
					got, want.cacheMicrosPer1M, want.src)
			}
		})
	}
}

// TestGeminiCostMicros_CachedIsDiscountedNotFree pins the cached-token
// semantics that CHO-2220 shipped with as a FLAGGED residual and this pass
// closes: cached input is billed at the model's cache rate, not at zero.
//
// Zero was never "no cost" — Vertex publishes real "... Input Caching" SKUs, so
// zero was an under-bill hiding as a simplification. It must also stay strictly
// CHEAPER than fresh input, or the discount has inverted.
func TestGeminiCostMicros_CachedIsDiscountedNotFree(t *testing.T) {
	for model, want := range vertexRates {
		t.Run(model, func(t *testing.T) {
			cached := geminiCostMicros(model, oneMillion, 0, oneMillion)
			fresh := geminiCostMicros(model, oneMillion, 0, 0)

			if cached == 0 {
				t.Errorf("fully-cached prompt bills 0 — cached tokens are discounted, not free "+
					"(%s)", want.src)
			}
			if cached >= fresh {
				t.Errorf("cached cost %d >= fresh-input cost %d — the cache discount has inverted",
					cached, fresh)
			}
		})
	}
}

// TestGeminiCostMicros_EveryPricedModelBillsNonZero is the regression guard for
// the defect class this story exists to kill: a priced model must never bill 0
// for real token counts. A typo that drops a model from the map, or zeroes a
// rate, reads as "free" everywhere downstream and is invisible without this.
func TestGeminiCostMicros_EveryPricedModelBillsNonZero(t *testing.T) {
	for model := range vertexRates {
		if got := geminiCostMicros(model, 10_000, 5_000, 0); got <= 0 {
			t.Errorf("%s: cost for 10k in / 5k out = %d micros, want > 0 "+
				"(a priced model billing 0 is the bug this story fixes)", model, got)
		}
	}
}

// TestGeminiCostMicros_UnpricedModelServesButWarnsLoudly encodes the owner's
// 2026-07-16 decision: an unpriced model keeps serving (refusing would turn an
// accounting gap into an inference outage) but the $0 must be LOUD, never
// silent. The prior code returned 0 with a comment claiming "downstream
// alerting can flag" it — no such alerting existed, which is exactly why the
// warn has to be asserted here.
func TestGeminiCostMicros_UnpricedModelServesButWarnsLoudly(t *testing.T) {
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewJSONHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn})))
	t.Cleanup(func() { slog.SetDefault(prev) })

	got := geminiCostMicros("gemini-9-imaginary", 1_000, 500, 0)
	if got != 0 {
		t.Fatalf("unpriced model cost = %d micros, want 0 (never guess a price)", got)
	}

	line := strings.TrimSpace(buf.String())
	if line == "" {
		t.Fatal("unpriced model produced NO log line — the $0 is silent, which is the bug")
	}
	var rec map[string]any
	if err := json.Unmarshal([]byte(strings.Split(line, "\n")[0]), &rec); err != nil {
		t.Fatalf("warn line is not JSON: %v (line=%q)", err, line)
	}
	if lvl, _ := rec["level"].(string); lvl != "WARN" {
		t.Errorf("unpriced model logged at level %q, want WARN", lvl)
	}
	if m, _ := rec["model"].(string); m != "gemini-9-imaginary" {
		t.Errorf("warn omits the offending model: model=%q, want %q", m, "gemini-9-imaginary")
	}
	// A stable boolean field is what a Cloud Logging log-based metric filters
	// on. Renaming it silently breaks the alert, so pin it.
	if flag, ok := rec["chora_unpriced_model"].(bool); !ok || !flag {
		t.Errorf("warn omits the stable chora_unpriced_model=true field a log-based metric keys on; got %v", rec["chora_unpriced_model"])
	}
}

// TestGeminiCostMicros_PreviewModelsAreDeliberatelyUnpriced documents a
// deliberate omission so a future reader does not "helpfully" invent rates.
//
//   - gemini-3.1-pro-preview has NO SKU in the Vertex catalogue at all (161
//     live calls in the 25d to 2026-07-16). $0 may be correct while it is
//     unbilled; it becomes wrong silently at GA — hence the loud warn.
//   - gemini-3-flash-preview HAS a "Gemini 3 Flash" GA SKU family ($0.50/$3.00)
//     but a preview model cannot be confirmed to bill under it. Pricing it
//     risks OVER-billing, which is fabrication in the other direction.
func TestGeminiCostMicros_PreviewModelsAreDeliberatelyUnpriced(t *testing.T) {
	for _, model := range []string{"gemini-3.1-pro-preview", "gemini-3-flash-preview"} {
		if _, priced := geminiPricing[model]; priced {
			t.Errorf("%s has a price entry — preview models must stay unpriced until a "+
				"SKU is confirmed; a guessed rate is fabricated billing data", model)
		}
	}
}

// TestGeminiCostMicros_ProImageScene prices one qgen scene image end-to-end.
// Gemini bills generated images as output tokens (~1290 per image), so the
// per-1M output rate has to land on a sane per-image figure. This is the number
// the owner accepted ("cost is acceptable for the increased quality") and it
// was previously recorded as $0.
func TestGeminiCostMicros_ProImageScene(t *testing.T) {
	const imageOutputTokens = 1290

	pro := geminiCostMicros("gemini-3-pro-image", 0, imageOutputTokens, 0)
	if want := int64(154_800); pro != want { // 1290 * 120_000_000 / 1e6
		t.Errorf("gemini-3-pro-image scene = %d micros ($%.4f), want %d micros ($%.4f)",
			pro, float64(pro)/1e6, want, float64(want)/1e6)
	}

	flash := geminiCostMicros("gemini-2.5-flash-image", 0, imageOutputTokens, 0)
	if want := int64(38_700); flash != want { // 1290 * 30_000_000 / 1e6
		t.Errorf("gemini-2.5-flash-image scene = %d micros, want %d micros", flash, want)
	}

	// The upgrade is a real 4x. If this ratio collapses, a rate is wrong.
	if pro <= flash {
		t.Errorf("gemini-3-pro-image (%d) should cost strictly more than gemini-2.5-flash-image (%d)", pro, flash)
	}
}

// TestGeminiCostMicros_CachedTokensAreNotDoubleCharged proves a cached token is
// billed ONCE, at the cache rate — never also at the fresh-input rate.
func TestGeminiCostMicros_CachedTokensAreNotDoubleCharged(t *testing.T) {
	const model = "gemini-2.5-flash"
	rate := vertexRates[model]

	// Half of a 1M-token prompt cached ⇒ half at fresh input, half at cache.
	got := geminiCostMicros(model, oneMillion, 0, oneMillion/2)
	want := rate.inMicrosPer1M/2 + rate.cacheMicrosPer1M/2
	if got != want {
		t.Errorf("half-cached prompt cost = %d micros, want %d (%d fresh + %d cached)",
			got, want, rate.inMicrosPer1M/2, rate.cacheMicrosPer1M/2)
	}

	// Caching must never cost MORE than not caching.
	if uncached := geminiCostMicros(model, oneMillion, 0, 0); got >= uncached {
		t.Errorf("half-cached (%d) >= fully-fresh (%d) — caching must be a discount", got, uncached)
	}

	// cachedTokens > promptTokens is nonsense the vendor should never send;
	// clamp rather than bill negative fresh input.
	if got := geminiCostMicros(model, 100, 0, 500); got < 0 {
		t.Errorf("cached > prompt produced negative cost %d; want clamped", got)
	}
}
