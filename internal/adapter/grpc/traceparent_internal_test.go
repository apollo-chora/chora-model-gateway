// traceparent_internal_test.go — unit tests for resolveTraceparent, the
// helper that guarantees every emitted TokenUsageEvent carries a valid,
// caller-correlated W3C traceparent.
//
// Regression context (HANDOFF_OBSERVABILITY_OUTBOX_JAM_2026-05-29 Fix 2):
// the shared publisher rejects an empty traceparent, and the model-gateway
// emitted events with none (callers propagated no trace context and there was
// no fallback). resolveTraceparent closes that: it prefers the active OTel
// span (set by the otelgrpc server handler — correlated to the caller), and
// falls back to tracing.EnsureTraceparent on the inbound string.
package modelgatewaygrpc

import (
	"context"
	"strings"
	"testing"

	"go.opentelemetry.io/otel/trace"
)

func TestResolveTraceparent_NoSpan_MintsValidWhenInboundEmpty(t *testing.T) {
	tp := resolveTraceparent(context.Background(), "")
	parts := strings.Split(tp, "-")
	if len(parts) != 4 || len(parts[1]) != 32 || len(parts[2]) != 16 {
		t.Fatalf("resolveTraceparent('') = %q; want a valid freshly-minted W3C traceparent", tp)
	}
	if strings.Trim(parts[1], "0") == "" {
		t.Errorf("minted trace_id is all-zero: %q", tp)
	}
}

func TestResolveTraceparent_NoSpan_PreservesValidInbound(t *testing.T) {
	inbound := "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
	if got := resolveTraceparent(context.Background(), inbound); got != inbound {
		t.Errorf("resolveTraceparent(valid inbound) = %q; want it preserved (%q)", got, inbound)
	}
}

func TestResolveTraceparent_PrefersActiveSpan(t *testing.T) {
	traceID, _ := trace.TraceIDFromHex("0af7651916cd43dd8448eb211c80319c")
	spanID, _ := trace.SpanIDFromHex("b7ad6b7169203331")
	sc := trace.NewSpanContext(trace.SpanContextConfig{
		TraceID:    traceID,
		SpanID:     spanID,
		TraceFlags: trace.FlagsSampled,
		Remote:     true,
	})
	ctx := trace.ContextWithSpanContext(context.Background(), sc)

	// Even with a DIFFERENT inbound string, the active span wins (it is the
	// gateway's own Invoke span, the correct causal parent).
	got := resolveTraceparent(ctx, "00-11111111111111111111111111111111-2222222222222222-01")
	want := "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
	if got != want {
		t.Errorf("resolveTraceparent(span ctx) = %q; want span-derived %q", got, want)
	}
}
