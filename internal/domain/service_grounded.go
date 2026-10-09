package domain

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"
)

// GroundedSearch executes the single controlled web-egress call (ADR-231),
// reusing the Invoke governance substance behind a citations-first surface.
//
// Order of operations (each a single port call — fail-closed, cheapest gates
// first so a broke/blocked tenant never triggers an external API cost):
//
//  1. Validate + resolve invocation_id.
//  2. ExternalEgressGate.Authorize — platform kill-switch + per-tenant
//     external_egress entitlement + daily ceiling. Fail-closed (error ⇒ deny).
//  3. BudgetRepo.GetTenantBudget + Decide — BudgetBlock ⇒ deny.
//  4. ManaMeter.Quote (fail-closed pre-flight) — not affordable ⇒ in-band
//     FinishReasonManaBlock (402 upsell), BEFORE any external cost.
//  5. PolicyLoader.ResolveAgentPolicy — Armor template + grounding model.
//  6. ArmorClient.SanitizeUserPrompt (PRE) on the directive — block ⇒ deny.
//  7. GroundedVendorClient.GroundedGenerate — the web egress.
//  8. ArmorClient.SanitizeModelResponse (POST) on the web-sourced answer —
//     block ⇒ deny (ledger still emitted; the vendor billed).
//  9. ManaMeter.Debit + ExternalEgressGate.RecordEgress + token-usage outbox.
//  10. ExternalEgressAudited event (ALWAYS — allowed OR denied) + return.
//
// Denials that prevent egress surface as *GroundedSearchError (the adapter maps
// to FAILED_PRECONDITION / PERMISSION_DENIED / RESOURCE_EXHAUSTED / UNAVAILABLE).
// Mana exhaustion + zero citations are IN-BAND outcomes (a GroundedSearchResult
// with a finish reason / empty citations) so the caller can render the upsell /
// hedge. Every path emits exactly one egress audit event.
func (s *Service) GroundedSearch(ctx context.Context, req GroundedSearchRequest) (GroundedSearchResult, error) {
	start := s.now()

	if err := s.groundedPortsWired(); err != nil {
		return GroundedSearchResult{}, err
	}
	if err := validateGroundedRequest(req); err != nil {
		return GroundedSearchResult{}, err
	}

	invocationID := req.InvocationID
	if invocationID == "" {
		invocationID = s.newID()
	}

	// Step 2 — external-egress gate (fail-closed). ANY error ⇒ deny (the
	// chokepoint does not trust an unreachable policy store).
	auth, err := s.egressGate.Authorize(ctx, req.TenantID)
	if err != nil {
		s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyDisabled, ArmorVerdictUnspecified, ArmorVerdictUnspecified, "", "", nil, 0)
		return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyDisabled, Detail: "external_egress gate unavailable (fail-closed)", Inner: err}
	}
	if !auth.Allowed {
		reason := auth.Reason
		if reason == "" {
			reason = EgressDenyDisabled
		}
		s.emitEgressAudit(ctx, req, EgressAuditDenied, reason, ArmorVerdictUnspecified, ArmorVerdictUnspecified, "", "", nil, 0)
		return GroundedSearchResult{}, &GroundedSearchError{Reason: reason, Detail: "external egress not permitted for tenant"}
	}

	// Step 3 — tenant budget.
	budget, err := s.budget.GetTenantBudget(ctx, req.TenantID)
	if err != nil {
		s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyBudget, ArmorVerdictUnspecified, ArmorVerdictUnspecified, "", "", nil, 0)
		return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyBudget, Detail: "budget repo unavailable", Inner: err}
	}
	if budget == nil {
		// Fail-closed "budget required" mode: a missing active budget window
		// denies the egress — the absence of a policy must not restore
		// unlimited provider spending. The mode is global
		// (CHORA_LLM_BUDGET_REQUIRED) or per-tenant
		// (CHORA_LLM_BUDGET_REQUIRED_TENANTS); the decision is made per
		// request from req.TenantID.
		if budgetRequiredFor(s.budgetRequired, s.budgetRequiredTenants, req.TenantID) {
			s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyBudget, ArmorVerdictUnspecified, ArmorVerdictUnspecified, "", "", nil, 0)
			return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyBudget, Detail: "no active LLM budget window for tenant; budget required (CHORA_LLM_BUDGET_REQUIRED)"}
		}
	} else if budget.Decide() == BudgetBlock {
		s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyBudget, ArmorVerdictUnspecified, ArmorVerdictUnspecified, "", "", nil, 0)
		return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyBudget, Detail: "tenant LLM budget exhausted"}
	}

	// Step 4 — mana pre-flight (fail-closed). A broke learner is blocked BEFORE
	// the priciest call in the catalogue fires. In-band (402 upsell), not error.
	q, err := s.mana.Quote(ctx, req.GCID, req.TenantID, req.ActionCode)
	if err != nil {
		// A meter outage on the PRICIEST action fails closed: refuse rather than
		// grant a free web egress.
		s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyManaBlock, ArmorVerdictUnspecified, ArmorVerdictUnspecified, "", "", nil, 0)
		return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyManaBlock, Detail: "mana meter unavailable (fail-closed)", Inner: err}
	}
	if !q.UnknownAction && !q.Affordable {
		s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyManaBlock, ArmorVerdictUnspecified, ArmorVerdictUnspecified, "", "", nil, 0)
		return GroundedSearchResult{
			InvocationID:   invocationID,
			FinishReason:   FinishReasonManaBlock,
			FinishDetail:   fmt.Sprintf("insufficient_mana required=%d available=%d action=%s", q.RequiredUnits, q.AvailableUnits, req.ActionCode),
			LatencyMs:      elapsedMs(start, s.now()),
			CompletedAt:    s.now(),
			GatewayVersion: s.gatewayVersion,
		}, nil
	}

	// Step 5 — resolve policy (Armor template + grounding model).
	policy, err := s.policies.ResolveAgentPolicy(ctx, req.AgentID, req.CrewKind, s.groundedModel(req), nil)
	if err != nil {
		s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyVendorError, ArmorVerdictUnspecified, ArmorVerdictUnspecified, "", "", nil, 0)
		return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyVendorError, Detail: "policy resolve failed", Inner: err}
	}

	// Step 6 — Armor PRE on the directive (the adversarial surface, pre-egress).
	armorPre := ArmorVerdictBypassed
	directive := req.Directive
	if policy.ArmorTemplate != "" {
		armorPre, directive, err = s.armor.SanitizeUserPrompt(ctx, policy.ArmorTemplate, req.Directive)
		if err != nil {
			s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyArmorPre, ArmorVerdictUnspecified, ArmorVerdictUnspecified, "", "", nil, 0)
			return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyArmorPre, Detail: "model armor PRE call failed", Inner: err}
		}
		if armorPre.IsTerminalBlock() {
			s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyArmorPre, armorPre, ArmorVerdictUnspecified, "", "", nil, 0)
			return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyArmorPre, Detail: "model armor PRE blocked the directive"}
		}
	}

	// Step 7 — grounded web dispatch (the single controlled egress).
	vendorResp, err := s.grounded.GroundedGenerate(ctx, GroundedVendorRequest{
		LogicalModelID: s.groundedModel(req),
		Directive:      directive,
		MaxResults:     effectiveMaxResults(req.MaxResults),
		Traceparent:    req.Traceparent,
		Tracestate:     req.Tracestate,
		TenantID:       req.TenantID,
	})
	if err != nil {
		s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyVendorError, armorPre, ArmorVerdictUnspecified, "", "", nil, 0)
		return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyVendorError, Detail: "grounded vendor dispatch failed", Inner: err}
	}
	vendorName := string(s.grounded.Family())

	// Step 8 — Armor POST on the WEB-SOURCED answer (prompt-injection-via-page
	// defence). A block still emits the ledger (the vendor billed) + records the
	// egress (the web call happened) but does NOT charge the learner.
	armorPost := ArmorVerdictBypassed
	if policy.ArmorTemplate != "" && vendorResp.Answer != "" {
		var perr error
		armorPost, _, perr = s.armor.SanitizeModelResponse(ctx, policy.ArmorTemplate, vendorResp.Answer)
		if perr != nil {
			s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyArmorPost, armorPre, ArmorVerdictUnspecified, vendorName, vendorResp.ModelVersion, vendorResp.WebSearchQueries, 0)
			return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyArmorPost, Detail: "model armor POST call failed", Inner: perr}
		}
		if armorPost.IsTerminalBlock() {
			s.debitBudget(ctx, budget, req.TenantID, vendorResp.Usage.CostMicros)
			_ = s.egressGate.RecordEgress(ctx, req.TenantID)
			s.emitGroundedOutbox(ctx, invocationID, req, vendorName, vendorResp, armorPre, armorPost)
			s.emitEgressAudit(ctx, req, EgressAuditDenied, EgressDenyArmorPost, armorPre, armorPost, vendorName, vendorResp.ModelVersion, vendorResp.WebSearchQueries, 0)
			return GroundedSearchResult{}, &GroundedSearchError{Reason: EgressDenyArmorPost, Detail: "model armor POST blocked the grounded answer"}
		}
	}

	// Step 9 — success accounting: budget debit + mana debit + daily-counter
	// increment + token-usage ledger.
	s.debitBudget(ctx, budget, req.TenantID, vendorResp.Usage.CostMicros)
	if _, derr := s.mana.Debit(ctx, req.GCID, req.TenantID, req.ActionCode, invocationID); derr != nil {
		// Served-but-uncharged: never un-serve a completed egress (resilience
		// over strictness, matching the Invoke mana decorator).
		_ = derr
	}
	if rerr := s.egressGate.RecordEgress(ctx, req.TenantID); rerr != nil {
		_ = rerr // daily counter is a soft cost rail; a miss must not un-serve.
	}
	s.emitGroundedOutbox(ctx, invocationID, req, vendorName, vendorResp, armorPre, armorPost)

	// Step 10 — build result + audit. Zero renderable citations is an ANOMALY
	// (flagged) but a legitimate in-band outcome — the CALLER hedges (ADR-220 D3).
	citations := vendorResp.Citations
	result := GroundedSearchResult{
		InvocationID:         invocationID,
		Citations:            citations,
		GroundedAnswer:       vendorResp.Answer,
		SearchEntryPointHTML: vendorResp.SearchEntryPointHTML,
		WebSearchQueries:     vendorResp.WebSearchQueries,
		Usage:                vendorResp.Usage,
		Vendor:               vendorName,
		ModelVersion:         vendorResp.ModelVersion,
		ArmorPre:             armorPre,
		ArmorPost:            armorPost,
		LatencyMs:            elapsedMs(start, s.now()),
		FinishReason:         FinishReasonComplete,
		CompletedAt:          s.now(),
		GatewayVersion:       s.gatewayVersion,
	}
	citeCount := renderableCitationCount(citations)
	auditResult := EgressAuditAllowed
	denial := ""
	if citeCount == 0 {
		auditResult = EgressAuditAnomaly
		denial = EgressDenyZeroCite
	}
	s.emitEgressAudit(ctx, req, auditResult, denial, armorPre, armorPost, vendorName, vendorResp.ModelVersion, vendorResp.WebSearchQueries, citeCount)
	return result, nil
}

// groundedPortsWired fails loud when GroundedSearch is invoked on an Invoke-only
// Service (the four ADR-231 ports were not provided at construction).
func (s *Service) groundedPortsWired() error {
	if s.egressGate == nil || s.grounded == nil || s.mana == nil || s.egressAudit == nil {
		return &GroundedSearchError{Reason: "not_wired", Detail: "GroundedSearch ports not wired (egressGate/grounded/mana/egressAudit)"}
	}
	return nil
}

func validateGroundedRequest(req GroundedSearchRequest) error {
	switch {
	case req.TenantID == "":
		return &GroundedSearchError{Reason: "invalid_argument", Detail: "tenant_id required"}
	case req.GCID == "":
		return &GroundedSearchError{Reason: "invalid_argument", Detail: "gcid required"}
	case req.AgentID == "":
		return &GroundedSearchError{Reason: "invalid_argument", Detail: "agent_id required"}
	case req.Directive == "":
		return &GroundedSearchError{Reason: "invalid_argument", Detail: "directive required"}
	case req.ActionCode == "":
		return &GroundedSearchError{Reason: "invalid_argument", Detail: "action_code required (un-priced egress not permitted)"}
	}
	return nil
}

// groundedModel resolves the grounding-capable model (request override or the
// ADR-231 D6 default). Grounding requires a grounding-capable model, so a
// budget downgrade to a cheaper non-grounding model is NOT applied here.
func (s *Service) groundedModel(req GroundedSearchRequest) LogicalModelID {
	if req.LogicalModelID != "" {
		return req.LogicalModelID
	}
	return DefaultGroundedModel
}

// effectiveMaxResults clamps the caller's citation cap into [1, hard cap],
// defaulting when unset.
func effectiveMaxResults(n int32) int32 {
	if n <= 0 {
		return DefaultMaxResults
	}
	if n > MaxCitationsHardCap {
		return MaxCitationsHardCap
	}
	return n
}

// renderableCitationCount counts citations satisfying the mandate (URI+Title).
func renderableCitationCount(cs []GroundedCitation) int32 {
	var n int32
	for _, c := range cs {
		if c.URI != "" && c.Title != "" {
			n++
		}
	}
	return n
}

func (s *Service) debitBudget(ctx context.Context, budget *BudgetState, tenantID string, costMicros int64) {
	if budget == nil || costMicros <= 0 {
		return
	}
	// Best-effort like Invoke's post-success debit; a debit failure must not
	// un-serve a completed egress (the outbox ledger is the canonical record).
	_ = s.budget.DebitSpent(ctx, tenantID, costMicros)
}

// emitGroundedOutbox writes the token-usage ledger row for a grounded call onto
// the shared observability outbox (same canonical topic as Invoke).
func (s *Service) emitGroundedOutbox(ctx context.Context, invocationID string, req GroundedSearchRequest, vendor string, vr GroundedVendorResponse, armorPre, armorPost ArmorVerdict) {
	evt := TokenUsageEvent{
		UsageID:        invocationID,
		TenantID:       req.TenantID,
		GCID:           req.GCID,
		ModelID:        vr.ModelVersion,
		InputTokens:    vr.Usage.InputTokens,
		OutputTokens:   vr.Usage.OutputTokens,
		CachedTokens:   vr.Usage.CachedTokens,
		CostMicros:     vr.Usage.CostMicros,
		InvocationID:   invocationID,
		AgentRole:      req.AgentID,
		AgentID:        req.AgentID,
		ActionCode:     actionCodeOrAgent(req.ActionCode, req.AgentID),
		RecordedAt:     s.now(),
		Vendor:         vendor,
		FallbackChain:  []string{fmt.Sprintf("%s:%s", vendor, vr.ModelVersion)},
		ArmorPre:       armorPre,
		ArmorPost:      armorPost,
		GatewayVersion: s.gatewayVersion,
		Traceparent:    req.Traceparent,
		Tracestate:     req.Tracestate,
	}
	// Ledger durability is best-effort at the domain layer (as with Invoke's
	// short-circuit paths); the adapter records failures on its OTel span.
	_ = s.outbox.EnqueueTokenUsageRecorded(ctx, evt)
}

// emitEgressAudit publishes exactly one ExternalEgressAudited event per grounded
// call (ADR-231 D6) — allowed OR denied. The raw directive is HASHED, never
// persisted. Armor verdicts ride as string tokens ("" when the stage was not
// reached).
func (s *Service) emitEgressAudit(
	ctx context.Context,
	req GroundedSearchRequest,
	result EgressAuditResult,
	denialReason string,
	armorPre, armorPost ArmorVerdict,
	vendor, modelVersion string,
	queries []string,
	citationCount int32,
) {
	sum := sha256.Sum256([]byte(req.Directive))
	evt := ExternalEgressAuditEvent{
		AuditID:          s.newID(),
		TenantID:         req.TenantID,
		ActorGCID:        req.GCID,
		AgentID:          req.AgentID,
		ActionCode:       req.ActionCode,
		DirectiveHash:    hex.EncodeToString(sum[:]),
		WebSearchQueries: queries,
		CitationCount:    citationCount,
		Result:           result,
		DenialReason:     denialReason,
		ArmorVerdictPre:  armorAuditToken(armorPre),
		ArmorVerdictPost: armorAuditToken(armorPost),
		Vendor:           vendor,
		ModelVersion:     modelVersion,
		OccurredAt:       s.now(),
		Traceparent:      req.Traceparent,
		Tracestate:       req.Tracestate,
	}
	// Audit is a first-class trail but a publish miss must not un-serve; the
	// adapter records the failure on its OTel span.
	_ = s.egressAudit.EnqueueExternalEgressAudited(ctx, evt)
}

// armorAuditToken maps an ArmorVerdict to its audit string token, using "" for
// the Unspecified zero value (the stage did not run) rather than "unspecified".
func armorAuditToken(v ArmorVerdict) string {
	if v == ArmorVerdictUnspecified {
		return ""
	}
	return v.String()
}

// elapsedMs returns the bounded milliseconds between two timestamps for
// LatencyMs (overflows int32 only past ~24.8 days — impossible for one call).
func elapsedMs(start, end time.Time) int32 {
	return int32(end.Sub(start).Milliseconds()) // #nosec G115 -- elapsed ms for one request; far below int32
}
