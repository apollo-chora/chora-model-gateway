package domain

import (
	"slices"
	"strings"
	"testing"
)

// TestNumberFrom covers every numeric type a generation config can legitimately
// carry, and — more importantly — what it must refuse.
//
// The values arrive from two places with different Go types: JSON decoding
// produces float64 for every number, while the hand-built maps on the HTTP and
// gRPC facades can carry int, int32, int64 or float32. Rejecting one of those
// would silently drop a sampling knob the caller asked for.
func TestNumberFrom(t *testing.T) {
	accepted := []struct {
		name string
		in   any
		want float64
	}{
		{"float64 from json", float64(0.75), 0.75},
		{"float32", float32(0.5), 0.5},
		{"int", int(64), 64},
		{"int32", int32(32), 32},
		{"int64", int64(1 << 40), 1 << 40},
		// Zero is a legitimate value and must survive; deciding "absent"
		// by way of a zero check would confuse max_tokens=0 with unset.
		{"zero float", float64(0), 0},
		{"zero int", int(0), 0},
		{"negative temperature is a value, not an error", float64(-1), -1},
	}
	for _, tc := range accepted {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := numberFrom(tc.in)
			if !ok {
				t.Fatalf("numberFrom(%T) refused a numeric value", tc.in)
			}
			if got != tc.want {
				t.Errorf("numberFrom(%T) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

// TestNumberFrom_RejectsNonNumbers is the fail-safe half. A string is not a
// number: coercing "0.5" or even "" into a float would ship a knob the caller
// never set, and — for max_tokens — a coerced garbage value becomes a silent
// generation ceiling.
func TestNumberFrom_RejectsNonNumbers(t *testing.T) {
	cases := []struct {
		name string
		in   any
	}{
		{"string", "0.5"},
		{"empty string", ""},
		{"bool", true},
		{"nil", nil},
		{"slice", []any{1}},
		{"map", map[string]any{"n": 1}},
		// json.Number is not in the switch. It is not produced by
		// encoding/json's default decoding into `any`, so refusing it is
		// correct — but the gap is deliberate-looking, so it is pinned here.
		{"struct", struct{ N int }{N: 1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, ok := numberFrom(tc.in)
			if ok {
				t.Errorf("numberFrom(%#v) = %v, true; want it refused", tc.in, got)
			}
			if got != 0 {
				t.Errorf("numberFrom(%#v) returned %v with ok=false, want the zero value", tc.in, got)
			}
		})
	}
}

// TestApplyOutputCeiling covers the only caller of numberFrom. The gate exists
// because a caller asking for more output than the model allows is a
// configuration mistake worth correcting silently — but a caller asking for
// less, or for something unparseable, must be left alone.
func TestApplyOutputCeiling(t *testing.T) {
	cases := []struct {
		name string
		cfg  map[string]any
		// want is the value max_tokens must carry afterwards, and wantPresent
		// says whether the key must still be there at all.
		want        float64
		wantPresent bool
	}{
		{"over the ceiling is clamped", map[string]any{"max_tokens": float64(9000)}, 4096, true},
		{"exactly the ceiling is left alone", map[string]any{"max_tokens": float64(4096)}, 4096, true},
		{"under the ceiling is left alone", map[string]any{"max_tokens": float64(10)}, 10, true},
		{"an int max_tokens is compared too", map[string]any{"max_tokens": 9000}, 4096, true},
		// Nothing to clamp: the key is not a number, so the gateway cannot tell
		// whether it exceeds the ceiling. Leaving it for the vendor to reject is
		// better than guessing a ceiling onto a value it never understood.
		{"an unparseable max_tokens is left alone", map[string]any{"max_tokens": "lots"}, 0, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out := applyOutputCeiling(tc.cfg, 4096)
			got, present := out["max_tokens"]
			if !tc.wantPresent || !present {
				t.Fatalf("max_tokens present = %v, want %v (map: %v)", present, tc.wantPresent, out)
			}
			f, isNum := numberFrom(got)
			if !isNum && tc.want == 0 {
				// The unparseable case: the key survives verbatim rather than
				// being replaced by a number the caller never sent.
				return
			}
			if !isNum || f != tc.want {
				t.Errorf("max_tokens = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestApplyOutputCeiling_AbsentMaxTokensIsNotInvented(t *testing.T) {
	// A caller who sent no max_tokens must not come back with one: the adapter
	// already has a required-field default for the Anthropic surface, and a
	// ceiling injected here would fight it.
	out := applyOutputCeiling(map[string]any{"temperature": 0.2}, 4096)
	if _, present := out["max_tokens"]; present {
		t.Errorf("max_tokens was invented: %v", out)
	}
}

func TestApplyOutputCeiling_NoCeilingIsANoOp(t *testing.T) {
	// Zero means "unconstrained" on a TargetModel, so the ceiling must not be
	// applied at all — otherwise an entry that never set max_output_tokens
	// would clamp every caller to zero output.
	cfg := map[string]any{"max_tokens": float64(9000)}
	out := applyOutputCeiling(cfg, 0)
	if got, _ := numberFrom(out["max_tokens"]); got != 9000 {
		t.Errorf("max_tokens = %v, want it untouched with no ceiling", got)
	}
}

func TestApplyOutputCeiling_NilConfigIsANoOp(t *testing.T) {
	if out := applyOutputCeiling(nil, 4096); out != nil {
		t.Errorf("applyOutputCeiling(nil, 4096) = %v, want nil", out)
	}
}

func TestApplyOutputCeiling_DoesNotMutateTheCallersMap(t *testing.T) {
	// The request map is shared with the caller and re-used across a fallback
	// chain; writing the clamp back into it would apply one target's ceiling to
	// the next dispatch in the chain.
	cfg := map[string]any{"max_tokens": float64(9000), "temperature": 0.5}
	out := applyOutputCeiling(cfg, 4096)
	if got, _ := numberFrom(out["max_tokens"]); got != 4096 {
		t.Errorf("returned max_tokens = %v, want the clamp", got)
	}
	if got, _ := numberFrom(cfg["max_tokens"]); got != 9000 {
		t.Errorf("the caller's map was mutated: max_tokens = %v, want 9000", got)
	}
}

// TestKnownCapabilities pins the vocabulary Validate checks against. A
// capability name that is a typo in a registry file must surface at boot; if it
// were absent from this list it would instead silently disable a gate — the
// model would advertise nothing it can do and the failure would appear as a
// confusing 4xx at dispatch time.
func TestKnownCapabilities(t *testing.T) {
	want := []string{
		CapabilityChat,
		CapabilityTools,
		CapabilityVision,
		CapabilityImage,
		CapabilityEmbeddings,
		CapabilityWebSearch,
	}
	if len(KnownCapabilities) != len(want) {
		t.Fatalf("KnownCapabilities = %v, want %d entries", KnownCapabilities, len(want))
	}
	seen := map[string]bool{}
	for _, c := range want {
		if !slices.Contains(KnownCapabilities, c) {
			t.Errorf("KnownCapabilities is missing %q", c)
		}
		if seen[c] {
			t.Errorf("KnownCapabilities lists %q twice", c)
		}
		seen[c] = true
	}
	if strings.Contains(strings.Join(KnownCapabilities, ","), " ") {
		t.Errorf("KnownCapabilities = %v, want lowercase single-word tokens", KnownCapabilities)
	}
}

// TestModalityConstants pins the modality vocabulary. The gRPC surface passes
// these strings through verbatim from the proto enum, so the spelling is a wire
// contract rather than an internal detail.
func TestModalityConstants(t *testing.T) {
	if ModalityText != "TEXT" || ModalityImage != "IMAGE" {
		t.Errorf("modality constants = %q / %q, want TEXT / IMAGE", ModalityText, ModalityImage)
	}
	// IMAGE is only reachable on a surface that advertises the image
	// capability; the pairing is what TargetModel.Supports gates on.
	if CapabilityImage != "image" {
		t.Errorf("CapabilityImage = %q, want the lowercase token the registry files use", CapabilityImage)
	}
}
