package domain_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strings"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ----------------------------------------------------------------------------
// GroundedSearch test fakes (ADR-231). Reuse fakeArmor / fakeBudget /
// fakeOutbox / fakePolicy from service_test.go; add the grounded-only ports.
// ----------------------------------------------------------------------------

type fakeEgressGate struct {
	auth        domain.EgressAuthorization
	authErr     error
	authCalls   int
	recordCalls int
	recordErr   error
}

func (f *fakeEgressGate) Authorize(ctx context.Context, tenantID string) (domain.EgressAuthorization, error) {
	f.authCalls++
	if f.authErr != nil {
		return domain.EgressAuthorization{}, f.authErr
	}
	return f.auth, nil
}

func (f *fakeEgressGate) RecordEgress(ctx context.Context, tenantID string) error {
	f.recordCalls++
	return f.recordErr
}

type fakeGroundedVendor struct {
	family  domain.VendorFamily
	resp    domain.GroundedVendorResponse
	err     error
	calls   int
	lastReq domain.GroundedVendorRequest
}

func (f *fakeGroundedVendor) Family() domain.VendorFamily { return f.family }

func (f *fakeGroundedVendor) GroundedGenerate(ctx context.Context, req domain.GroundedVendorRequest) (domain.GroundedVendorResponse, error) {
	f.calls++
	f.lastReq = req
	if f.err != nil {
		return domain.GroundedVendorResponse{}, f.err
	}
	return f.resp, nil
}

type fakeMana struct {
	quote         domain.ManaQuote
	quoteErr      error
	debit         domain.ManaDebit
	debitErr      error
	quoteCalls    int
	debitCalls    int
	lastDebitIdem string
	lastAction    string
}

func (f *fakeMana) Quote(ctx context.Context, gcid, tenantID, actionCode string) (domain.ManaQuote, error) {
	f.quoteCalls++
	f.lastAction = actionCode
	if f.quoteErr != nil {
		return domain.ManaQuote{}, f.quoteErr
	}
	return f.quote, nil
}

func (f *fakeMana) Debit(ctx context.Context, gcid, tenantID, actionCode, idemKey string) (domain.ManaDebit, error) {
	f.debitCalls++
	f.lastDebitIdem = idemKey
	if f.debitErr != nil {
		return domain.ManaDebit{}, f.debitErr
	}
	return f.debit, nil
}

type fakeEgressAudit struct {
	events []domain.ExternalEgressAuditEvent
	err    error
}

func (f *fakeEgressAudit) EnqueueExternalEgressAudited(ctx context.Context, evt domain.ExternalEgressAuditEvent) error {
	if f.err != nil {
		return f.err
	}
	f.events = append(f.events, evt)
	return nil
}

// groundedFakes bundles every port fake so a test can assert on any of them.
type groundedFakes struct {
	egress   *fakeEgressGate
	grounded *fakeGroundedVendor
	mana     *fakeMana
	armor    *fakeArmor
	budget   *fakeBudget
	outbox   *fakeOutbox
	audit    *fakeEgressAudit
	policy   *fakePolicy
}

// newGroundedTestService wires a Service with happy-path defaults for all
// grounded ports. mods tweak the config for a specific case.
func newGroundedTestService(t *testing.T, mods ...func(f *groundedFakes)) (*domain.Service, *groundedFakes) {
	t.Helper()

	f := &groundedFakes{
		egress: &fakeEgressGate{auth: domain.EgressAuthorization{Allowed: true}},
		grounded: &fakeGroundedVendor{
			family: domain.VendorFamilyVertexGemini,
			resp: domain.GroundedVendorResponse{
				Answer: "The water cycle moves water via evaporation, condensation, precipitation.",
				Citations: []domain.GroundedCitation{
					{URI: "https://redirect/1", Title: "Water cycle", Domain: "nasa.gov", Snippet: "evaporation...", Confidence: 0.9},
					{URI: "https://redirect/2", Title: "Precipitation", Domain: "noaa.gov", Snippet: "condensation...", Confidence: 0.8},
				},
				SearchEntryPointHTML: "<div class=\"gsc\">chip</div>",
				WebSearchQueries:     []string{"water cycle stages"},
				Usage:                domain.TokenUsage{InputTokens: 40, OutputTokens: 120, CostMicros: 9_000},
				ModelVersion:         "gemini-2.5-flash",
				FinishReason:         domain.FinishReasonComplete,
			},
		},
		mana:   &fakeMana{quote: domain.ManaQuote{Affordable: true, RequiredUnits: 80, AvailableUnits: 500}, debit: domain.ManaDebit{Success: true, BalanceAfterUnits: 420}},
		armor:  &fakeArmor{preVerdict: domain.ArmorVerdictAllow, postVerdict: domain.ArmorVerdictAllow},
		budget: &fakeBudget{state: &domain.BudgetState{TenantID: "t", BudgetUSDMicros: 1_000_000, SpentUSDMicros: 0, Policy: domain.BudgetPolicyBlock}},
		outbox: &fakeOutbox{},
		audit:  &fakeEgressAudit{},
		policy: &fakePolicy{policy: domain.AgentPolicy{
			AgentID:                "familiar_seeker",
			ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-flash"),
			Vendor:                 domain.VendorFamilyVertexGemini,
			ArmorTemplate:          "projects/chora-489812/locations/asia-southeast1/templates/chora-guardrail-balanced-dev",
		}},
	}
	for _, m := range mods {
		m(f)
	}

	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{&fakeVendor{family: domain.VendorFamilyVertexGemini}},
		Armor:          f.armor,
		Budget:         f.budget,
		Outbox:         f.outbox,
		Policies:       f.policy,
		GatewayVersion: "chora-model-gateway:test",
		EgressGate:     f.egress,
		Grounded:       f.grounded,
		Mana:           f.mana,
		EgressAudit:    f.audit,
		Now:            fixedTime,
		NewID:          func() string { return "gs-invocation-id" },
	})
	require.NoError(t, err)
	return svc, f
}

func groundedReq() domain.GroundedSearchRequest {
	return domain.GroundedSearchRequest{
		TenantID:    "11111111-1111-7111-8111-111111111111",
		GCID:        "22222222-2222-7222-8222-222222222222",
		AgentID:     "familiar_seeker",
		CrewKind:    "familiar",
		Directive:   "Is the Great Wall of China visible from space?",
		ActionCode:  "familiar_far_sight_grounded_search",
		Traceparent: "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
	}
}

// ----------------------------------------------------------------------------
// Happy path.
// ----------------------------------------------------------------------------

func TestGroundedSearch_HappyPath_CitedResult(t *testing.T) {
	svc, f := newGroundedTestService(t)

	res, err := svc.GroundedSearch(context.Background(), groundedReq())
	require.NoError(t, err)

	// Result carries citations + answer + chip + queries.
	assert.Len(t, res.Citations, 2)
	assert.True(t, res.HasCitations())
	assert.Equal(t, "nasa.gov", res.Citations[0].Domain)
	assert.NotEmpty(t, res.GroundedAnswer)
	assert.Equal(t, "<div class=\"gsc\">chip</div>", res.SearchEntryPointHTML)
	assert.Equal(t, []string{"water cycle stages"}, res.WebSearchQueries)
	assert.Equal(t, domain.FinishReasonComplete, res.FinishReason)
	assert.Equal(t, domain.ArmorVerdictAllow, res.ArmorPre)
	assert.Equal(t, domain.ArmorVerdictAllow, res.ArmorPost)
	assert.Equal(t, "gs-invocation-id", res.InvocationID)
	assert.Equal(t, "vertex_ai_gemini", res.Vendor)

	// Chain executed in the right shape.
	assert.Equal(t, 1, f.egress.authCalls, "egress authorized once")
	assert.Equal(t, 1, f.grounded.calls, "grounded vendor dispatched once")
	assert.Equal(t, 1, f.mana.quoteCalls, "mana pre-flight quoted")
	assert.Equal(t, 1, f.mana.debitCalls, "mana debited post-success")
	assert.Equal(t, "gs-invocation-id", f.mana.lastDebitIdem, "debit idempotent on invocation id")
	assert.Equal(t, "familiar_far_sight_grounded_search", f.mana.lastAction)
	assert.Equal(t, 1, f.egress.recordCalls, "daily counter incremented")
	require.Len(t, f.outbox.events, 1, "token-usage ledger emitted")

	// The directive was Armor-screened + forwarded (screened text) to the vendor.
	assert.Equal(t, 1, f.armor.preCalls)
	assert.Equal(t, 1, f.armor.postCalls)
	assert.Equal(t, "Is the Great Wall of China visible from space?", f.grounded.lastReq.Directive)
	assert.Equal(t, domain.DefaultMaxResults, f.grounded.lastReq.MaxResults)

	// Egress audit ALLOWED with a HASHED directive (never the raw text).
	require.Len(t, f.audit.events, 1)
	ev := f.audit.events[0]
	assert.Equal(t, domain.EgressAuditAllowed, ev.Result)
	assert.Empty(t, ev.DenialReason)
	assert.Equal(t, int32(2), ev.CitationCount)
	assert.Equal(t, []string{"water cycle stages"}, ev.WebSearchQueries)
	assert.Equal(t, "familiar_seeker", ev.AgentID)
	sum := sha256.Sum256([]byte("Is the Great Wall of China visible from space?"))
	assert.Equal(t, hex.EncodeToString(sum[:]), ev.DirectiveHash)
	assert.NotContains(t, ev.DirectiveHash, "Great Wall", "raw directive must not be persisted")
	assert.Equal(t, "allow", ev.ArmorVerdictPre)
	assert.Equal(t, "allow", ev.ArmorVerdictPost)
}

func TestGroundedSearch_HardCapsMaxResults(t *testing.T) {
	svc, f := newGroundedTestService(t)
	req := groundedReq()
	req.MaxResults = 50 // above the hard cap
	_, err := svc.GroundedSearch(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, domain.MaxCitationsHardCap, f.grounded.lastReq.MaxResults)
}

// ----------------------------------------------------------------------------
// External-egress gate — the fail-closed governance surface.
// ----------------------------------------------------------------------------

func TestGroundedSearch_EgressDisabled_DeniesBeforeAnyCost(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.egress.auth = domain.EgressAuthorization{Allowed: false, Reason: domain.EgressDenyDisabled}
	})

	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyDisabled, gerr.Reason)

	// No egress, no mana, no vendor — denied before any cost.
	assert.Equal(t, 0, f.grounded.calls)
	assert.Equal(t, 0, f.mana.quoteCalls)
	assert.Equal(t, 0, f.mana.debitCalls)
	assert.Equal(t, 0, f.egress.recordCalls)
	assert.Empty(t, f.outbox.events)

	// But the denial IS audited.
	require.Len(t, f.audit.events, 1)
	assert.Equal(t, domain.EgressAuditDenied, f.audit.events[0].Result)
	assert.Equal(t, domain.EgressDenyDisabled, f.audit.events[0].DenialReason)
}

func TestGroundedSearch_KillSwitch_Denies(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.egress.auth = domain.EgressAuthorization{Allowed: false, Reason: domain.EgressDenyKillSwitch}
	})
	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyKillSwitch, gerr.Reason)
	assert.Equal(t, 0, f.grounded.calls)
	require.Len(t, f.audit.events, 1)
	assert.Equal(t, domain.EgressDenyKillSwitch, f.audit.events[0].DenialReason)
}

func TestGroundedSearch_EgressGateError_FailsClosed(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.egress.authErr = errors.New("policy store unreachable")
	})
	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	require.Error(t, err, "an unreachable policy store must fail CLOSED, not open")
	assert.Equal(t, 0, f.grounded.calls)
}

func TestGroundedSearch_DailyCeiling_Denies(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.egress.auth = domain.EgressAuthorization{Allowed: false, Reason: domain.EgressDenyDailyCeil}
	})
	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyDailyCeil, gerr.Reason)
	assert.Equal(t, 0, f.grounded.calls)
}

// ----------------------------------------------------------------------------
// Budget + mana gates.
// ----------------------------------------------------------------------------

func TestGroundedSearch_BudgetBlock_Denies(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.budget.state = &domain.BudgetState{TenantID: "t", BudgetUSDMicros: 100, SpentUSDMicros: 100, Policy: domain.BudgetPolicyBlock}
	})
	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyBudget, gerr.Reason)
	assert.Equal(t, 0, f.grounded.calls)
	require.Len(t, f.audit.events, 1)
	assert.Equal(t, domain.EgressAuditDenied, f.audit.events[0].Result)
}

func TestGroundedSearch_ManaInsufficient_InBandBlock(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.mana.quote = domain.ManaQuote{Affordable: false, RequiredUnits: 80, AvailableUnits: 10}
	})

	res, err := svc.GroundedSearch(context.Background(), groundedReq())
	require.NoError(t, err, "mana block flows IN-BAND (402 upsell), not as an error")
	assert.Equal(t, domain.FinishReasonManaBlock, res.FinishReason)
	assert.Contains(t, res.FinishDetail, "insufficient_mana")
	assert.Contains(t, res.FinishDetail, "required=80")
	assert.Contains(t, res.FinishDetail, "available=10")

	// Fail-closed BEFORE the expensive egress: no vendor, no debit, no record.
	assert.Equal(t, 0, f.grounded.calls)
	assert.Equal(t, 0, f.mana.debitCalls)
	assert.Equal(t, 0, f.egress.recordCalls)
	require.Len(t, f.audit.events, 1)
	assert.Equal(t, domain.EgressDenyManaBlock, f.audit.events[0].DenialReason)
}

// ----------------------------------------------------------------------------
// Armor PRE / POST.
// ----------------------------------------------------------------------------

func TestGroundedSearch_ArmorPreBlock_NoEgress(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.armor.preVerdict = domain.ArmorVerdictBlock
	})
	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyArmorPre, gerr.Reason)
	assert.Equal(t, 0, f.grounded.calls, "Armor PRE block ⇒ no web call")
	assert.Equal(t, 0, f.mana.debitCalls)
	require.Len(t, f.audit.events, 1)
	assert.Equal(t, "block", f.audit.events[0].ArmorVerdictPre)
}

func TestGroundedSearch_ArmorPostBlock_EgressHappenedNoManaDebit(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.armor.postVerdict = domain.ArmorVerdictBlock
	})
	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyArmorPost, gerr.Reason)

	// The web call DID fire (cost tracked) but the blocked answer isn't charged.
	assert.Equal(t, 1, f.grounded.calls)
	assert.Equal(t, 0, f.mana.debitCalls, "no mana charge for a blocked answer")
	require.Len(t, f.outbox.events, 1, "vendor billed ⇒ ledger emitted")
	require.Len(t, f.audit.events, 1)
	assert.Equal(t, domain.EgressAuditDenied, f.audit.events[0].Result)
	assert.Equal(t, "block", f.audit.events[0].ArmorVerdictPost)
}

// ----------------------------------------------------------------------------
// Zero citations + vendor error.
// ----------------------------------------------------------------------------

func TestGroundedSearch_ZeroCitations_InBandHedge(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.grounded.resp.Citations = nil
	})
	res, err := svc.GroundedSearch(context.Background(), groundedReq())
	require.NoError(t, err, "zero citations is an in-band outcome; the CALLER hedges")
	assert.False(t, res.HasCitations())
	assert.Empty(t, res.Citations)
	assert.Equal(t, domain.FinishReasonComplete, res.FinishReason)

	// Egress happened → recorded + charged; audited as an ANOMALY.
	assert.Equal(t, 1, f.egress.recordCalls)
	assert.Equal(t, 1, f.mana.debitCalls)
	require.Len(t, f.audit.events, 1)
	assert.Equal(t, domain.EgressAuditAnomaly, f.audit.events[0].Result)
	assert.Equal(t, domain.EgressDenyZeroCite, f.audit.events[0].DenialReason)
	assert.Equal(t, int32(0), f.audit.events[0].CitationCount)
}

func TestGroundedSearch_VendorError_Surfaces(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.grounded.err = errors.New("vertex 503")
	})
	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyVendorError, gerr.Reason)
	assert.Equal(t, 0, f.mana.debitCalls, "no charge when the vendor failed")
	require.Len(t, f.audit.events, 1)
	assert.Equal(t, domain.EgressAuditDenied, f.audit.events[0].Result)
}

// ----------------------------------------------------------------------------
// Validation + unwired guards.
// ----------------------------------------------------------------------------

func TestGroundedSearch_ManaMeterError_FailsClosed(t *testing.T) {
	// A mana-meter OUTAGE on the priciest action must fail CLOSED — refuse,
	// never grant a free web egress (unlike the fail-OPEN Invoke default).
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.mana.quoteErr = errors.New("identity unreachable")
	})
	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyManaBlock, gerr.Reason)
	assert.Equal(t, 0, f.grounded.calls, "no web egress when the meter is down")
	require.Len(t, f.audit.events, 1)
	assert.Equal(t, domain.EgressAuditDenied, f.audit.events[0].Result)
}

func TestGroundedSearch_PolicyResolveError_Surfaces(t *testing.T) {
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.policy.err = errors.New("policy yaml missing")
	})
	_, err := svc.GroundedSearch(context.Background(), groundedReq())
	var gerr *domain.GroundedSearchError
	require.ErrorAs(t, err, &gerr)
	assert.Equal(t, domain.EgressDenyVendorError, gerr.Reason)
	assert.Equal(t, 0, f.grounded.calls)
}

func TestGroundedSearch_ModelOverride_ForwardedToVendor(t *testing.T) {
	svc, f := newGroundedTestService(t)
	req := groundedReq()
	req.LogicalModelID = "gemini-3-flash"
	_, err := svc.GroundedSearch(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, domain.LogicalModelID("gemini-3-flash"), f.grounded.lastReq.LogicalModelID)
}

func TestGroundedSearch_UnpricedAction_ProceedsUnmetered(t *testing.T) {
	// An unpriced action_code (catalogue miss) is un-metered — the egress still
	// proceeds (governance gates already passed); no phantom block.
	svc, f := newGroundedTestService(t, func(f *groundedFakes) {
		f.mana.quote = domain.ManaQuote{UnknownAction: true}
	})
	res, err := svc.GroundedSearch(context.Background(), groundedReq())
	require.NoError(t, err)
	assert.True(t, res.HasCitations())
	assert.Equal(t, 1, f.grounded.calls)
}

func TestGroundedSearch_ValidatesRequiredFields(t *testing.T) {
	svc, _ := newGroundedTestService(t)
	for _, tc := range []struct {
		name string
		mut  func(r *domain.GroundedSearchRequest)
	}{
		{"missing tenant", func(r *domain.GroundedSearchRequest) { r.TenantID = "" }},
		{"missing gcid", func(r *domain.GroundedSearchRequest) { r.GCID = "" }},
		{"missing agent", func(r *domain.GroundedSearchRequest) { r.AgentID = "" }},
		{"missing directive", func(r *domain.GroundedSearchRequest) { r.Directive = "" }},
		{"missing action_code", func(r *domain.GroundedSearchRequest) { r.ActionCode = "" }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r := groundedReq()
			tc.mut(&r)
			_, err := svc.GroundedSearch(context.Background(), r)
			require.Error(t, err)
		})
	}
}

func TestGroundedSearch_UnwiredPorts_FailLoud(t *testing.T) {
	// A Service built WITHOUT the grounded ports (Invoke-only wiring) must
	// refuse GroundedSearch loudly rather than nil-panic.
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{&fakeVendor{family: domain.VendorFamilyVertexGemini}},
		Armor:          &fakeArmor{},
		Budget:         &fakeBudget{},
		Outbox:         &fakeOutbox{},
		Policies:       &fakePolicy{},
		GatewayVersion: "chora-model-gateway:test",
		Now:            fixedTime,
		NewID:          func() string { return "x" },
	})
	require.NoError(t, err)
	_, err = svc.GroundedSearch(context.Background(), groundedReq())
	require.Error(t, err)
	assert.True(t, strings.Contains(err.Error(), "not wired") || strings.Contains(err.Error(), "unwired"))
}
