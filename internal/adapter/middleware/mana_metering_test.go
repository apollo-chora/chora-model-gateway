package middleware_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/middleware"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// --- fakes -------------------------------------------------------------------

type fakeNext struct {
	called int
	req    domain.InvokeRequest
	resp   domain.InvokeResponse
	err    error
}

func (f *fakeNext) Invoke(_ context.Context, req domain.InvokeRequest) (domain.InvokeResponse, error) {
	f.called++
	f.req = req
	return f.resp, f.err
}

type fakeMana struct {
	quote      middleware.ManaQuote
	quoteErr   error
	quoteCalls int

	debit      middleware.ManaDebit
	debitErr   error
	debitCalls int
	lastDebit  struct{ gcid, tenant, action, idem string }
}

func (f *fakeMana) Quote(_ context.Context, gcid, tenantID, actionCode string) (middleware.ManaQuote, error) {
	f.quoteCalls++
	return f.quote, f.quoteErr
}

func (f *fakeMana) Debit(_ context.Context, gcid, tenantID, actionCode, idemKey string) (middleware.ManaDebit, error) {
	f.debitCalls++
	f.lastDebit.gcid = gcid
	f.lastDebit.tenant = tenantID
	f.lastDebit.action = actionCode
	f.lastDebit.idem = idemKey
	return f.debit, f.debitErr
}

func testConfig() middleware.Config {
	c := middleware.DefaultConfig()
	c.Now = func() time.Time { return time.Unix(0, 0).UTC() }
	c.NewID = func() string { return "gen-invocation-id" }
	// R22: the keyed debit claim is taken by the domain service (reported on
	// InvokeResponse.Deduped); Config.Claims is retained but no longer used.
	return c
}

func baseReq() domain.InvokeRequest {
	return domain.InvokeRequest{
		InvocationID:   "inv-1",
		TenantID:       "11111111-1111-7111-8111-111111111111",
		GCID:           "00000000-0000-7000-8000-000000001999",
		AgentID:        "companion_chat",
		CrewKind:       "companion_chat",
		LogicalModelID: "gemini-2.5-flash",
		Prompt:         "hi",
	}
}

func okResp() domain.InvokeResponse {
	return domain.InvokeResponse{InvocationID: "inv-1", Completion: "hello", FinishReason: domain.FinishReasonComplete}
}

// --- tests -------------------------------------------------------------------

// Fail-closed action (companion_chat*) with sufficient balance: pre-flight
// passes, LLM runs, debit is taken post-success keyed by the invocation id.
func TestManaMetering_FailClosed_Affordable_CallsAndDebits(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}, debit: middleware.ManaDebit{Success: true}}
	req := baseReq()
	req.ActionCode = "companion_chat_turn_basic"

	m := middleware.NewManaMetering(next, mp, testConfig())
	resp, err := m.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if next.called != 1 {
		t.Errorf("wrapped called %d times, want 1", next.called)
	}
	if mp.quoteCalls != 1 {
		t.Errorf("Quote called %d times, want 1 (pre-flight)", mp.quoteCalls)
	}
	if mp.debitCalls != 1 {
		t.Errorf("Debit called %d times, want 1 (post-success)", mp.debitCalls)
	}
	if mp.lastDebit.action != "companion_chat_turn_basic" {
		t.Errorf("debit action = %q", mp.lastDebit.action)
	}
	if mp.lastDebit.idem != "inv-1" {
		t.Errorf("debit idem = %q, want the invocation id inv-1", mp.lastDebit.idem)
	}
	if resp.FinishReason != domain.FinishReasonComplete {
		t.Errorf("FinishReason = %v, want Complete (passed through)", resp.FinishReason)
	}
}

// Fail-closed action with insufficient balance: hard-block BEFORE calling the
// LLM; surface FinishReasonManaBlock + structured upsell; no debit.
func TestManaMetering_FailClosed_Insufficient_Blocks(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: false, RequiredUnits: 30, AvailableUnits: 5}}
	req := baseReq()
	req.ActionCode = "companion_chat_turn_premium"

	m := middleware.NewManaMetering(next, mp, testConfig())
	resp, err := m.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if next.called != 0 {
		t.Errorf("wrapped called %d times, want 0 (hard-blocked pre-LLM)", next.called)
	}
	if mp.debitCalls != 0 {
		t.Errorf("Debit called %d times, want 0 (blocked)", mp.debitCalls)
	}
	if resp.FinishReason != domain.FinishReasonManaBlock {
		t.Fatalf("FinishReason = %v, want ManaBlock", resp.FinishReason)
	}
	if !strings.Contains(resp.FinishDetail, "required=30") || !strings.Contains(resp.FinishDetail, "available=5") {
		t.Errorf("FinishDetail = %q, want required=30 available=5", resp.FinishDetail)
	}
}

// W3-to-W4 overlap: consumption still stamps the legacy familiar_chat_turn_*
// codes until its W4 cut (the chat binary passes the caller's mana_action_code
// through), so the legacy prefix must stay fail-closed beside companion_chat
// or a zero-balance learner gets those turns for free meanwhile.
func TestManaMetering_LegacyFamiliarChatPrefix_StaysFailClosed(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: false, RequiredUnits: 5, AvailableUnits: 0}}
	req := baseReq()
	req.AgentID = "familiar_companion"
	req.CrewKind = "familiar_companion"
	req.ActionCode = "familiar_chat_turn_basic"

	m := middleware.NewManaMetering(next, mp, testConfig())
	resp, err := m.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if next.called != 0 {
		t.Errorf("wrapped called %d times, want 0 (legacy chat code must hard-block pre-LLM during the overlap)", next.called)
	}
	if resp.FinishReason != domain.FinishReasonManaBlock {
		t.Fatalf("FinishReason = %v, want ManaBlock", resp.FinishReason)
	}
}

// Fail-open authoring action with insufficient balance: never blocks; LLM runs,
// post-debit shortfall is served (fail-open + audit).
func TestManaMetering_FailOpen_Insufficient_Serves(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{debit: middleware.ManaDebit{Success: false, RequiredUnits: 25, BalanceAfterUnits: 0}}
	req := baseReq()
	req.AgentID = "qgen_question"
	req.CrewKind = "qgen"
	req.ActionCode = "atom_authoring_assist"

	m := middleware.NewManaMetering(next, mp, testConfig())
	resp, err := m.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if mp.quoteCalls != 0 {
		t.Errorf("Quote called %d times, want 0 (fail-open never pre-flights)", mp.quoteCalls)
	}
	if next.called != 1 {
		t.Errorf("wrapped called %d times, want 1 (authoring never blocks)", next.called)
	}
	if mp.debitCalls != 1 {
		t.Errorf("Debit called %d times, want 1 (best-effort post-success)", mp.debitCalls)
	}
	if resp.FinishReason != domain.FinishReasonComplete {
		t.Errorf("FinishReason = %v, want Complete (served despite shortfall)", resp.FinishReason)
	}
}

// Unknown / unmapped action_code: un-metered passthrough, no debit.
func TestManaMetering_UnknownAction_Unmetered(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{}
	req := baseReq()
	req.ActionCode = ""
	req.AgentID = "content_moderation_probe" // not in fallback map
	req.CrewKind = "content_moderation"      // not in fallback map

	m := middleware.NewManaMetering(next, mp, testConfig())
	_, err := m.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if next.called != 1 {
		t.Errorf("wrapped called %d times, want 1", next.called)
	}
	if mp.quoteCalls != 0 || mp.debitCalls != 0 {
		t.Errorf("metered an unmapped action: quote=%d debit=%d, want 0/0", mp.quoteCalls, mp.debitCalls)
	}
}

// LLM error: no debit (no charge-then-fail).
func TestManaMetering_LLMError_NoDebit(t *testing.T) {
	next := &fakeNext{err: errors.New("vendor exploded")}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}}
	req := baseReq()
	req.ActionCode = "companion_chat_turn_basic"

	m := middleware.NewManaMetering(next, mp, testConfig())
	_, err := m.Invoke(context.Background(), req)
	if err == nil {
		t.Fatalf("expected the wrapped error to propagate")
	}
	if mp.debitCalls != 0 {
		t.Errorf("Debit called %d times on LLM error, want 0", mp.debitCalls)
	}
}

// Non-completion finish reasons (armor/budget block) are not debit-worthy.
func TestManaMetering_BlockedResponse_NoDebit(t *testing.T) {
	next := &fakeNext{resp: domain.InvokeResponse{InvocationID: "inv-1", FinishReason: domain.FinishReasonModelArmorBlock}}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}}
	req := baseReq()
	req.ActionCode = "companion_chat_turn_basic"

	m := middleware.NewManaMetering(next, mp, testConfig())
	_, err := m.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if mp.debitCalls != 0 {
		t.Errorf("Debit called %d times on armor-block response, want 0", mp.debitCalls)
	}
}

// Meter error on a fail-closed action, default (non-strict): proceed + serve
// (don't deny service on a meter outage). The DEFINITIVE shortfall is the only
// hard-block.
func TestManaMetering_FailClosed_MeterError_DefaultProceeds(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{quoteErr: errors.New("identity unavailable")}
	req := baseReq()
	req.ActionCode = "companion_chat_turn_basic"

	m := middleware.NewManaMetering(next, mp, testConfig())
	resp, err := m.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if next.called != 1 {
		t.Errorf("wrapped called %d times, want 1 (meter outage must not deny service by default)", next.called)
	}
	if resp.FinishReason != domain.FinishReasonComplete {
		t.Errorf("FinishReason = %v, want Complete", resp.FinishReason)
	}
}

// Meter error on a fail-closed action, strict-on-error: block.
func TestManaMetering_FailClosed_MeterError_StrictBlocks(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{quoteErr: errors.New("identity unavailable")}
	cfg := testConfig()
	cfg.StrictOnError = true
	req := baseReq()
	req.ActionCode = "companion_chat_turn_basic"

	m := middleware.NewManaMetering(next, mp, cfg)
	resp, err := m.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if next.called != 0 {
		t.Errorf("wrapped called %d times, want 0 (strict-on-error blocks)", next.called)
	}
	if resp.FinishReason != domain.FinishReasonManaBlock {
		t.Errorf("FinishReason = %v, want ManaBlock", resp.FinishReason)
	}
}

// Quote reporting UnknownAction (unpriced) proceeds un-metered (no debit).
func TestManaMetering_FailClosed_QuoteUnknownAction_Unmetered(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{quote: middleware.ManaQuote{UnknownAction: true}}
	req := baseReq()
	req.ActionCode = "companion_chat_turn_basic"

	m := middleware.NewManaMetering(next, mp, testConfig())
	_, err := m.Invoke(context.Background(), req)
	if err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if next.called != 1 {
		t.Errorf("wrapped called %d times, want 1", next.called)
	}
	if mp.debitCalls != 0 {
		t.Errorf("Debit called %d times on unpriced action, want 0", mp.debitCalls)
	}
}

// The agent_id/crew_kind fallback map resolves an action_code when the caller
// omits it, for the one crew NOT yet migrated to explicit per-turn action
// codes (daily_dose -> daily_dose_coach). The fog_orchestrator row was deleted
// at ADR-254 D13 (the fog orchestrator is retired; kg_explorer stamps
// knowledge_graph_traverse explicitly), so an unstamped fog Invoke resolves
// NOTHING here and is refused upstream by the surface gate.
func TestManaMetering_FallbackMap_ResolvesActionCode(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}, debit: middleware.ManaDebit{Success: true}}
	req := baseReq()
	req.AgentID = "daily_dose"
	req.CrewKind = "daily_dose"
	req.ActionCode = ""

	m := middleware.NewManaMetering(next, mp, testConfig())
	if _, err := m.Invoke(context.Background(), req); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if mp.debitCalls != 1 {
		t.Fatalf("Debit called %d times, want 1 (resolved via fallback)", mp.debitCalls)
	}
	if mp.lastDebit.action != "daily_dose_coach" {
		t.Errorf("resolved action = %q, want daily_dose_coach (daily_dose fallback)", mp.lastDebit.action)
	}
}

// ADR-254 D13: a fog_orchestrator Invoke with no action_code resolves no
// fallback any more; the metering layer leaves it un-metered (the surface
// gate ahead of it is what refuses the unstamped fog call).
func TestManaMetering_FallbackMap_FogRowDeletedAtD13(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}, debit: middleware.ManaDebit{Success: true}}
	req := baseReq()
	req.AgentID = "fog_orchestrator"
	req.CrewKind = "fog_orchestrator"
	req.ActionCode = ""

	m := middleware.NewManaMetering(next, mp, testConfig())
	if _, err := m.Invoke(context.Background(), req); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if mp.debitCalls != 0 {
		t.Fatalf("Debit called %d times, want 0 (no fog fallback after D13)", mp.debitCalls)
	}
}

// ADR-178 §2 — qgen/qgen_question are REMOVED from the FallbackMap so that
// turning the gateway umbrella meter on for the non-qgen crews does NOT
// collaterally meter qgen at the gateway. qgen stays on chora-creation's
// granular authoring economics (question_authoring_ai_draft / _model_answer /
// _batch_parse / _batch_per_item, with refund-on-failure) until the
// configurable price-plan rules layer ships (ADR-178). The W8 image-gen call
// rides the same qgen_question agent_id, so this removal also keeps image-gen
// off the gateway meter — its cost is already covered by the creation-side
// authoring debit. Regression guard: an omitted action_code from a qgen /
// qgen_question caller must NOT be re-derived a code (genuinely un-metered).
func TestManaMetering_FallbackMap_QgenRemoved_Unmetered(t *testing.T) {
	for _, id := range []string{"qgen", "qgen_question"} {
		t.Run(id, func(t *testing.T) {
			next := &fakeNext{resp: okResp()}
			mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}, debit: middleware.ManaDebit{Success: true}}
			req := baseReq()
			req.AgentID = id
			req.CrewKind = id
			req.ActionCode = ""

			m := middleware.NewManaMetering(next, mp, testConfig())
			if _, err := m.Invoke(context.Background(), req); err != nil {
				t.Fatalf("Invoke: %v", err)
			}
			if mp.quoteCalls != 0 {
				t.Errorf("Quote called %d times for %q, want 0 (un-metered — qgen removed from FallbackMap)", mp.quoteCalls, id)
			}
			if mp.debitCalls != 0 {
				t.Errorf("Debit called %d times for %q, want 0 (un-metered — qgen removed from FallbackMap)", mp.debitCalls, id)
			}
			if next.called != 1 {
				t.Errorf("wrapped called %d times for %q, want 1 (passthrough)", next.called, id)
			}
		})
	}
}

// ADR-178 / FU-4(b) — meter_home=creation invariant at the gateway-config level.
// qgen authoring is metered by chora-creation (granular + refund-on-failure);
// now that question_authoring_* are first-class priced actions in the price-plan
// layer, a stray FallbackMap entry mapping an agent to one of them — or a qgen-ish
// key — would re-introduce the double-meter. Lock the default config so neither
// the keys nor the values of the FallbackMap reference any creation-homed
// authoring action code.
func TestManaMetering_DefaultFallbackMap_NoCreationHomedAuthoringCodes(t *testing.T) {
	forbidden := map[string]struct{}{
		"qgen":                              {},
		"qgen_question":                     {},
		"question_generation":               {},
		"question_authoring_ai_draft":       {},
		"question_authoring_model_answer":   {},
		"question_authoring_batch_parse":    {},
		"question_authoring_batch_per_item": {},
	}
	fm := middleware.DefaultConfig().FallbackMap
	for k, v := range fm {
		if _, bad := forbidden[k]; bad {
			t.Errorf("FallbackMap KEY %q is a qgen/authoring code — meter_home=creation; gateway must not derive it", k)
		}
		if _, bad := forbidden[v]; bad {
			t.Errorf("FallbackMap maps %q → %q, a creation-homed authoring code — would double-meter qgen at the gateway", k, v)
		}
	}
}

// ADR-177 contract — an umbrella-managed crew (the Learning Companion chat agent) that routes through the
// gateway and deliberately sends an EMPTY action_code (tool-result continuation,
// or a not-yet-cut-over interaction like greet/exam-prep) must NOT be re-derived
// a code via the fallback map: it is genuinely un-metered. Regression guard for
// the over-/double-charge bug fixed by removing familiar/familiar_companion (now companion_chat, ADR-254 D7: no renamed or new agent id ever enters the FallbackMap) from
// the FallbackMap.
func TestManaMetering_EmptyActionCode_CompanionNotFallbackMetered(t *testing.T) {
	next := &fakeNext{resp: okResp()}
	mp := &fakeMana{quote: middleware.ManaQuote{Affordable: true}, debit: middleware.ManaDebit{Success: true}}
	req := baseReq() // AgentID companion_chat / CrewKind companion_chat
	req.ActionCode = ""

	m := middleware.NewManaMetering(next, mp, testConfig())
	if _, err := m.Invoke(context.Background(), req); err != nil {
		t.Fatalf("Invoke: %v", err)
	}
	if mp.quoteCalls != 0 {
		t.Errorf("Quote called %d times, want 0 (un-metered — no fallback for companion_chat)", mp.quoteCalls)
	}
	if mp.debitCalls != 0 {
		t.Errorf("Debit called %d times, want 0 (un-metered — no fallback for companion_chat)", mp.debitCalls)
	}
}
