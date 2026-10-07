// Package middleware holds Invoke decorators for chora-model-gateway.
//
// ManaMetering is the WS-1 umbrella-metering seam (ADR-142 §4): every LLM call
// through the gateway debits one per-GCID mana wallet by action_code. Mana is
// the umbrella prepaid currency for ALL LLM calls (authoring AI-assist +
// Familiar + learner consumption all drain one wallet).
//
// Flow per Invoke:
//  1. Resolve the action_code — explicit InvokeRequest.ActionCode, else a
//     derivation from AgentID / CrewKind via the fallback map. Empty (unmapped)
//     ⇒ un-metered passthrough (logged once).
//  2. Classify fail-open vs fail-closed:
//     - fail-closed (premium learner: companion_chat*, knowledge_graph_traverse,
//     boss_challenge_atom_gen, daily_dose_coach, coach_session_full):
//     PRE-FLIGHT a dry-run affordability check; a DEFINITIVE shortfall
//     hard-blocks BEFORE spending LLM compute (in-band FinishReasonManaBlock
//     + upsell). A meter ERROR (identity unreachable) proceeds by default —
//     a meter outage must not deny service — unless StrictOnError.
//     - fail-open (authoring / instructor / free-tier, the default): NEVER
//     blocks; debits best-effort post-success and serves on shortfall.
//  3. Call the wrapped Invoker (the LLM).
//  4. Post-success (only when the LLM actually produced output —
//     FinishReasonComplete / MaxTokens): debit the catalogue cost (units=0)
//     idempotent on the invocation id (retries collapse to one ledger row). A
//     debit error / shortfall after a served call is logged, never un-serves
//     (no charge-then-fail; resilience over strictness). A DISPATCHED turn
//     (InvokeRequest.DispatchIdempotencyKey set) first takes a keyed claim on
//     (gcid, key, action_code) through the DebitClaimer (ADR-254 D7, R22): a
//     redelivered dispatch finds the key claimed and bills nothing.
//
// The ADR-252 containment read is NOT here: it is a separate decorator
// (CompanionSuspension) wired AHEAD of this one so a contained turn is refused
// before the mana Quote (deny-before-debit).
//
// Mana costs + the action catalogue live in chora_identity.mana_action_pricing;
// this decorator carries NO prices — it speaks action_codes to the ManaPort.
package middleware

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// Next is the wrapped invoker (domain.Service or another decorator).
type Next interface {
	Invoke(ctx context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error)
}

// ManaQuote is the result of a dry-run affordability check.
type ManaQuote struct {
	Affordable     bool
	RequiredUnits  int64
	AvailableUnits int64
	// UnknownAction is true when the action_code is not in the pricing
	// catalogue (unpriced ⇒ treat as un-metered).
	UnknownAction bool
}

// ManaDebit is the result of an actual debit.
type ManaDebit struct {
	Success           bool
	RequiredUnits     int64
	BalanceAfterUnits int64
	UnknownAction     bool
}

// DebitClaimer is the R22 keyed-claim port (ADR-254 D7): before debiting a
// DISPATCHED turn (InvokeRequest.DispatchIdempotencyKey set), the seam claims
// (gcid, dispatch_idempotency_key, action_code). claimed=true means this call
// won the key and must debit; claimed=false means the key was already billed
// by an earlier delivery (debit_deduped=true) and the call proceeds unbilled.
// Implemented by adapter/pg.Repo.ClaimDebit (chora_observability migration 0019).
type DebitClaimer interface {
	ClaimDebit(ctx context.Context, gcid, dispatchKey, actionCode, invocationID string) (bool, error)
}

// ManaPort is the gateway-side port to the identity ManaService. Implemented by
// adapter/clients.ManaClient (gRPC).
type ManaPort interface {
	// Quote runs a dry-run affordability check for action_code WITHOUT
	// debiting. UnknownAction ⇒ the code is unpriced.
	Quote(ctx context.Context, gcid, tenantID, actionCode string) (ManaQuote, error)
	// Debit charges the catalogue cost for action_code (units=0), idempotent on
	// idemKey. UnknownAction ⇒ unpriced (no charge).
	Debit(ctx context.Context, gcid, tenantID, actionCode, idemKey string) (ManaDebit, error)
}

// Config configures ManaMetering. Defaults via DefaultConfig; main.go overrides
// from env (no inline config — feedback_no_inline_config).
type Config struct {
	// FallbackMap derives an action_code from a lower-cased AgentID or CrewKind
	// when the caller omits ActionCode.
	FallbackMap map[string]string
	// FailClosedExact + FailClosedPrefix classify premium-learner actions that
	// hard-block on a definitive shortfall.
	FailClosedExact  map[string]struct{}
	FailClosedPrefix []string
	// StrictOnError blocks a fail-closed action when the meter ERRORS (can't
	// verify affordability). Default false: a meter outage proceeds + logs.
	StrictOnError bool

	// Claims is the R22 keyed debit claimer (ADR-254 D7). DEPRECATED: the
	// domain service now takes the claim internally and reports the result
	// on InvokeResponse.Deduped. The middleware reads Deduped instead of
	// taking its own claim. This field is retained for backward
	// compatibility but is no longer used.
	Claims DebitClaimer

	Now    func() time.Time
	NewID  func() string
	Logger *slog.Logger
}

// DefaultConfig returns the canonical metering policy.
func DefaultConfig() Config {
	return Config{
		FallbackMap: map[string]string{
			// crew_kind / agent_id → catalogue action_code, for crews that do
			// NOT yet send an explicit per-turn action_code.
			//
			// ⚠ ADR-177 contract: a crew that routes through the gateway and
			// sets its action_code EXPLICITLY per turn (Familiar, recommender,
			// moderation) MUST NOT appear here. The umbrella client deliberately
			// blanks action_code on tool-result continuations (one debit per
			// user-facing turn) and leaves it empty for un-metered interactions
			// (Familiar greet/exam-prep until their cutover); a fallback entry
			// would re-derive a code on exactly those blanked calls and
			// over-charge (every tool-loop iteration) or double-charge (greet/
			// exam-prep also carry a consumption debit). `familiar`/
			// `familiar_companion` were removed for this reason (review 2026-06-04).
			//
			// ⚠ ADR-178 §2: qgen / qgen_question are REMOVED. Metering qgen at
			// the gateway as flat question_generation (50) would re-price
			// authoring 5–10× (creation meters granular ai_draft=10 /
			// model_answer=5 / batch_parse=50 / batch_per_item=5×N with
			// refund-on-failure), un-meter batch, and lose refund-on-failure.
			// qgen STAYS on chora-creation's granular economics until the
			// configurable price-plan rules layer (H+) ships. Removing them
			// here is the pre-condition that lets the umbrella meter turn ON
			// for the non-qgen crews (familiar/recommender/moderation) without
			// collaterally double-metering qgen. The W8 image-gen call rides
			// the same qgen_question agent_id (image_client.py), so this also
			// keeps image-gen off the gateway meter — its cost is already
			// folded into the creation-side authoring debit.
			//
			// daily_dose stays until it migrates to explicit per-turn action
			// codes (it has no gateway-metered crew yet). The fog_orchestrator
			// row was DELETED at ADR-254 D13 (2026-08-22, coordinator signal):
			// the fog orchestrator is retired and kg_explorer stamps
			// knowledge_graph_traverse explicitly; no renamed or new agent id
			// ever enters this map again (ADR-254 D7).
			"daily_dose": "daily_dose_coach",
		},
		FailClosedExact: map[string]struct{}{
			"knowledge_graph_traverse": {},
			"boss_challenge_atom_gen":  {},
			"daily_dose_coach":         {},
			// coach_session_full (500 mana, multi-turn exam-prep coach) is the
			// most expensive learner action — included beyond the handoff's
			// named list so a zero-balance learner cannot get it for free.
			"coach_session_full": {},
		},
		// R17 / ADR-254 D7: the Learning Companion chat turn codes
		// (companion_chat_turn_{basic,standard,premium}) hard-block on a
		// definitive shortfall. Renamed from familiar_chat in the action-code
		// cut, which lands identity's pricing rows, consumption's cost_map and
		// the agent resolver in the SAME window as this prefix.
		// W3-to-W4 overlap: the legacy familiar_chat prefix stays beside
		// companion_chat because consumption keeps stamping
		// familiar_chat_turn_{tier} until its W4 cut (the chat binary passes
		// the caller's mana_action_code through). Without it those turns
		// would be served fail-OPEN to a zero-balance learner for the whole
		// window. Drop "familiar_chat" on the gateway roll after the W4
		// consumption cut (ADR-254 D7 / R17).
		FailClosedPrefix: []string{"companion_chat", "familiar_chat"},
		Now:              time.Now,
		Logger:           slog.Default(),
	}
}

// ManaMetering decorates an Invoker with per-GCID mana metering.
type ManaMetering struct {
	next Next
	mana ManaPort
	cfg  Config
}

// NewManaMetering wires the decorator. next + mana MUST be non-nil
// (feedback_no_stubs_real_wiring); a nil cfg field falls back to a default.
// The Claims config is no longer required — the domain service takes the
// keyed debit claim internally and reports the result on
// InvokeResponse.Deduped.
func NewManaMetering(next Next, mana ManaPort, cfg Config) *ManaMetering {
	if next == nil {
		panic("middleware.NewManaMetering: next Invoker required")
	}
	if mana == nil {
		panic("middleware.NewManaMetering: ManaPort required")
	}
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &ManaMetering{next: next, mana: mana, cfg: cfg}
}

// Invoke implements the gRPC adapter's Invoker.
func (m *ManaMetering) Invoke(ctx context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error) {
	actionCode := m.resolveActionCode(req)
	if actionCode == "" {
		// Unmapped → un-metered passthrough (e.g. content_moderation safety call).
		m.cfg.Logger.InfoContext(ctx, "mana metering: un-metered LLM call (no action_code)",
			"agent_id", req.AgentID, "crew_kind", req.CrewKind, "gcid", req.GCID)
		return m.next.Invoke(ctx, req)
	}
	failClosed := m.failClosed(actionCode)

	// Pre-flight gate — fail-closed only (fail-open never blocks).
	if failClosed {
		q, err := m.mana.Quote(ctx, req.GCID, req.TenantID, actionCode)
		switch {
		case err != nil:
			if m.cfg.StrictOnError {
				m.cfg.Logger.WarnContext(ctx, "mana metering: meter error on fail-closed action → BLOCK (strict)",
					"action_code", actionCode, "gcid", req.GCID, "err", err)
				return m.block(req, actionCode, 0, 0), nil
			}
			// Resilience: a meter outage must not deny service. Proceed +
			// best-effort post-debit (which will also error, logged there).
			m.cfg.Logger.WarnContext(ctx, "mana metering: meter error on fail-closed action → PROCEED (non-strict)",
				"action_code", actionCode, "gcid", req.GCID, "err", err)
		case q.UnknownAction:
			// Unpriced → un-metered.
			m.cfg.Logger.InfoContext(ctx, "mana metering: unpriced action_code → un-metered",
				"action_code", actionCode, "gcid", req.GCID)
			return m.next.Invoke(ctx, req)
		case !q.Affordable:
			m.cfg.Logger.InfoContext(ctx, "mana metering: insufficient mana → BLOCK",
				"action_code", actionCode, "gcid", req.GCID,
				"required", q.RequiredUnits, "available", q.AvailableUnits)
			return m.block(req, actionCode, q.RequiredUnits, q.AvailableUnits), nil
		}
	}

	// Call the LLM.
	resp, err := m.next.Invoke(ctx, req)
	if err != nil {
		return resp, err // LLM error → no debit (no charge-then-fail).
	}
	if !debitWorthy(resp.FinishReason) {
		// Armor/budget/mana/vendor in-band block — the model produced no
		// billable output.
		return resp, nil
	}

	// Post-success debit — idempotent on the invocation id.
	idem := resp.InvocationID
	if idem == "" {
		idem = req.InvocationID
	}
	if idem == "" && m.cfg.NewID != nil {
		idem = m.cfg.NewID()
	}

	// R22 (ADR-254 D7): the domain service takes the keyed debit claim
	// (gcid, dispatch_idempotency_key, action_code) BEFORE any spend and
	// reports the result on InvokeResponse.Deduped. A redelivered dispatch
	// (Deduped=true) skips the mana debit — one claim suppresses BOTH the
	// tenant-budget debit and the mana debit.
	if resp.Deduped {
		m.cfg.Logger.InfoContext(ctx, "mana metering: redelivered dispatch, debit_deduped=true",
			"action_code", actionCode, "gcid", req.GCID, "idem", idem, "debit_deduped", true)
		return resp, nil
	}

	d, derr := m.mana.Debit(ctx, req.GCID, req.TenantID, actionCode, idem)
	switch {
	case derr != nil:
		// Served but uncharged (meter error post-success). Log loudly; never
		// un-serve a completed call.
		m.cfg.Logger.ErrorContext(ctx, "mana metering: post-success debit error — served-but-uncharged",
			"action_code", actionCode, "gcid", req.GCID, "idem", idem, "fail_closed", failClosed, "err", derr)
	case d.UnknownAction:
		m.cfg.Logger.InfoContext(ctx, "mana metering: unpriced action_code at debit → un-metered",
			"action_code", actionCode, "gcid", req.GCID)
	case !d.Success:
		// Shortfall after a served call: fail-open authoring deficit, or a rare
		// fail-closed race past the pre-flight (concurrent spend). Either way the
		// answer was produced — serve + audit, never un-serve.
		m.cfg.Logger.WarnContext(ctx, "mana metering: post-success debit shortfall — served + audited",
			"action_code", actionCode, "gcid", req.GCID, "fail_closed", failClosed,
			"required", d.RequiredUnits, "balance_after", d.BalanceAfterUnits)
	}
	return resp, nil
}

// resolveActionCode returns the action_code to meter, or "" when un-metered.
func (m *ManaMetering) resolveActionCode(req domain.InvokeRequest) string {
	if c := strings.TrimSpace(req.ActionCode); c != "" {
		return c
	}
	if c, ok := m.cfg.FallbackMap[strings.ToLower(strings.TrimSpace(req.AgentID))]; ok {
		return c
	}
	if c, ok := m.cfg.FallbackMap[strings.ToLower(strings.TrimSpace(req.CrewKind))]; ok {
		return c
	}
	return ""
}

// failClosed reports whether an action_code hard-blocks on shortfall.
func (m *ManaMetering) failClosed(actionCode string) bool {
	for _, p := range m.cfg.FailClosedPrefix {
		if strings.HasPrefix(actionCode, p) {
			return true
		}
	}
	_, ok := m.cfg.FailClosedExact[actionCode]
	return ok
}

// block builds the in-band mana-block response (FinishReasonManaBlock) with the
// structured upsell the FE 402 modal renders.
func (m *ManaMetering) block(req domain.InvokeRequest, actionCode string, required, available int64) domain.InvokeResponse {
	id := req.InvocationID
	if id == "" && m.cfg.NewID != nil {
		id = m.cfg.NewID()
	}
	return domain.InvokeResponse{
		InvocationID: id,
		FinishReason: domain.FinishReasonManaBlock,
		FinishDetail: fmt.Sprintf("insufficient_mana required=%d available=%d action=%s", required, available, actionCode),
		CompletedAt:  m.cfg.Now(),
	}
}

// debitWorthy reports whether a finish reason represents billable LLM output.
func debitWorthy(fr domain.FinishReason) bool {
	return fr == domain.FinishReasonComplete || fr == domain.FinishReasonMaxTokens
}
