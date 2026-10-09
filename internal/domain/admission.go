package domain

import (
	"fmt"
	"sync"
)

// =============================================================================
// Tenant concurrency ceiling — the smallest enforceable cost bound.
//
// The tenant budget check is a read-then-debit that is NOT atomic: several
// in-flight invocations can all read the same remaining budget and all pass.
// A per-tenant in-flight cap bounds that window, so the maximum overshoot past
// an exhausted budget is (limit × worst-case cost of one invocation) instead of
// unbounded.
//
// SINGLE-INSTANCE ASSUMPTION — READ BEFORE ENABLING:
// This limiter is PROCESS-LOCAL. It is a correct ceiling only when every
// billable request for the capped tenant reaches exactly ONE gateway process.
// If the tenant's traffic can reach several replicas, the effective ceiling is
// (limit × replicas) and this limiter under-counts. Either pin the tenant to a
// single replica / single gateway Deployment, or replace this with shared
// admission (a DB or Redis counter). Do not enable it on a multi-replica
// deployment and treat the number as global.
// =============================================================================

// TenantConcurrencyLimiter caps the number of simultaneously in-flight billable
// model invocations per tenant. A nil *TenantConcurrencyLimiter is valid and
// means "unlimited": Acquire always succeeds and returns a no-op release, so
// the disabled path costs nothing and needs no branch at the call site.
type TenantConcurrencyLimiter struct {
	limit  int
	mu     sync.Mutex
	active map[string]int
}

// NewTenantConcurrencyLimiter builds a limiter with the given per-tenant cap.
// A limit <= 0 returns nil — the unlimited/off default that preserves the
// historical behaviour.
func NewTenantConcurrencyLimiter(limit int) *TenantConcurrencyLimiter {
	if limit <= 0 {
		return nil
	}
	return &TenantConcurrencyLimiter{limit: limit, active: make(map[string]int)}
}

// Limit reports the configured per-tenant cap; 0 when unlimited.
func (l *TenantConcurrencyLimiter) Limit() int {
	if l == nil {
		return 0
	}
	return l.limit
}

// Acquire takes one slot for tenantID. It returns a release function and true
// when a slot was available; false (and a nil release) when the tenant is
// already at its ceiling. The caller MUST call release once per successful
// Acquire — typically via defer, so the slot is returned on every exit path
// including a panic. A repeated call on the same release is a no-op, so a
// defensive double-release cannot free another caller's slot.
//
// A nil receiver always succeeds, which is how the "off" configuration is
// expressed without a branch at the call site.
func (l *TenantConcurrencyLimiter) Acquire(tenantID string) (release func(), ok bool) {
	if l == nil {
		return func() {}, true
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.active[tenantID] >= l.limit {
		return nil, false
	}
	l.active[tenantID]++
	var once sync.Once
	return func() {
		once.Do(func() {
			l.mu.Lock()
			defer l.mu.Unlock()
			if l.active[tenantID] <= 1 {
				delete(l.active, tenantID)
				return
			}
			l.active[tenantID]--
		})
	}, true
}

// OverloadedError is the typed refusal returned when a tenant is at its
// concurrency ceiling. It is a 429/RESOURCE_EXHAUSTED-class outcome, NOT a
// vendor failure: the request never reached a provider, so nothing was billed.
// The adapters map it to RESOURCE_EXHAUSTED (gRPC) / 429 (HTTP).
type OverloadedError struct {
	TenantID string
	Limit    int
}

func (e *OverloadedError) Error() string {
	return fmt.Sprintf("tenant %s is at its concurrency ceiling (%d concurrent billable invocations); retry later", e.TenantID, e.Limit)
}

// overloadDetail renders the operator-facing message for an OverloadedError,
// carrying the actionable numbers (the configured cap and the fix).
func overloadDetail(tenantID string, limit int) string {
	return fmt.Sprintf("tenant concurrency limit reached (tenant=%s limit=%d); retry later or raise CHORA_LLM_MAX_CONCURRENT_PER_TENANT", tenantID, limit)
}
