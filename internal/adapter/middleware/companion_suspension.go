// companion_suspension.go: the ADR-252 D1 containment decorator (ADR-254 D7).
//
// An O+ operator can contain the Learning Companion without a deploy: platform-
// wide, per tenant, or per skill. The control is enforced HERE, at the single
// un-bypassable LLM chokepoint, AHEAD of ManaMetering (today the mana Quote is
// the outermost gate), so the deny lands before any debit:
//
//  1. Surface: an ABSENT InvokeRequest.surface is a LOUD condition (ADR-252 Q1):
//     refuse with surface_unstamped. Never "not a companion turn".
//  2. Non-companion surfaces (qgen, oe_grading, ...) are not governed: pass
//     through without a read.
//  3. Companion surfaces (companion_chat, companion_diagnosis): read the two
//     chora_observability tables UNCACHED, per request, through the
//     SuspensionGate port. Any engaged row matching (platform | this tenant)
//     x (all skills | this action_code) denies with companion_suspended.
//  4. Fail CLOSED on a read error (suspension_unreadable); absent rows ALLOW.
//     The two are told apart by error, never by an empty result (ADR-252 D6).
//
// Denials are counted and stamped on the active span (ADR-252 D4: per-denied-
// turn audit rows were rejected as flood); the operator WRITE is what gets the
// hash-chained audit event, from chora-observability.
package middleware

import (
	"context"
	"fmt"
	"log/slog"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// SuspensionVerdict is the result of one uncached containment read.
type SuspensionVerdict struct {
	Suspended bool
	// Scope is "platform" or "tenant" when Suspended.
	Scope string
	// SkillKey is the matched skill (empty when the suspension covers all skills).
	SkillKey string
	// Reason is the operator's mandatory reason on the matched row.
	Reason string
}

// SuspensionGate is the gateway-side port over the two ADR-252 tables
// (platform_companion_suspension, companion_suspension_policy). Implemented by
// adapter/pg.Repo.CheckSuspension. A non-nil error means the state could not be
// read (the decorator denies); an empty store is (false, nil).
type SuspensionGate interface {
	CheckSuspension(ctx context.Context, tenantID, actionCode string) (SuspensionVerdict, error)
}

// SuspensionConfig configures the decorator. Zero values are filled with defaults.
type SuspensionConfig struct {
	Logger *slog.Logger
	// Denied is called once per refused turn with the reason token, for the
	// operator-facing counter (ADR-252 D4). Optional; nil means no counter.
	Denied func(ctx context.Context, reason, surface, tenantID string)
}

// CompanionSuspension decorates an Invoker with the containment read.
type CompanionSuspension struct {
	next Next
	gate SuspensionGate
	cfg  SuspensionConfig
}

// NewCompanionSuspension wires the decorator. next + gate MUST be non-nil
// (feedback_no_stubs_real_wiring): an unwired gate would be a control that
// silently does not exist.
func NewCompanionSuspension(next Next, gate SuspensionGate, cfg SuspensionConfig) *CompanionSuspension {
	if next == nil {
		panic("middleware.NewCompanionSuspension: next Invoker required")
	}
	if gate == nil {
		panic("middleware.NewCompanionSuspension: SuspensionGate required")
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &CompanionSuspension{next: next, gate: gate, cfg: cfg}
}

// Invoke implements Next / the gRPC adapter's Invoker.
func (c *CompanionSuspension) Invoke(ctx context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error) {
	if req.Surface == "" {
		c.deny(ctx, domain.DenySurfaceUnstamped, req)
		return domain.InvokeResponse{}, &domain.PreconditionError{
			Reason: domain.DenySurfaceUnstamped,
			Detail: fmt.Sprintf("InvokeRequest.surface is required (agent_id=%s crew_kind=%s)", req.AgentID, req.CrewKind),
		}
	}
	if !domain.IsCompanionSurface(req.Surface) {
		return c.next.Invoke(ctx, req)
	}

	v, err := c.gate.CheckSuspension(ctx, req.TenantID, req.ActionCode)
	if err != nil {
		c.deny(ctx, domain.DenySuspensionUnreadable, req)
		return domain.InvokeResponse{}, &domain.PreconditionError{
			Reason: domain.DenySuspensionUnreadable,
			Detail: "companion containment state unreadable; refusing (fail closed)",
			Inner:  err,
		}
	}
	if v.Suspended {
		c.deny(ctx, domain.DenyCompanionSuspended, req)
		skill := v.SkillKey
		if skill == "" {
			skill = "all"
		}
		return domain.InvokeResponse{}, &domain.PreconditionError{
			Reason: domain.DenyCompanionSuspended,
			Detail: fmt.Sprintf("scope=%s skill=%s reason=%q", v.Scope, skill, v.Reason),
		}
	}
	return c.next.Invoke(ctx, req)
}

func (c *CompanionSuspension) deny(ctx context.Context, reason string, req domain.InvokeRequest) {
	span := trace.SpanFromContext(ctx)
	span.SetAttributes(
		attribute.String("chora.companion.deny_reason", reason),
		attribute.String("chora.invoke.surface", req.Surface),
		attribute.String("chora.tenant_id", req.TenantID),
	)
	c.cfg.Logger.WarnContext(ctx, "companion suspension: turn refused",
		"reason", reason, "surface", req.Surface, "tenant_id", req.TenantID,
		"gcid", req.GCID, "agent_id", req.AgentID, "action_code", req.ActionCode)
	if c.cfg.Denied != nil {
		c.cfg.Denied(ctx, reason, req.Surface, req.TenantID)
	}
}
