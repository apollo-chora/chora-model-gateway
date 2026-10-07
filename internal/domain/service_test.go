package domain_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ----------------------------------------------------------------------------
// Test fakes — minimal in-memory implementations of each port.
//
// Per feedback_no_stubs_real_wiring these fakes are for UNIT testing only —
// they live in service_test.go (test-only build tag implicit via _test.go).
// Real adapters land in Phase 2.2 (vendor) + Phase 2.3 (Armor / Budget /
// Outbox / Secrets) and replace these in production wiring.
// ----------------------------------------------------------------------------

type fakeVendor struct {
	family  domain.VendorFamily
	resp    domain.VendorResponse
	err     error
	calls   int
	lastReq domain.VendorRequest
}

func (f *fakeVendor) Family() domain.VendorFamily { return f.family }

func (f *fakeVendor) Generate(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	f.calls++
	f.lastReq = req
	if f.err != nil {
		return domain.VendorResponse{}, f.err
	}
	return f.resp, nil
}

type fakeArmor struct {
	preVerdict      domain.ArmorVerdict
	preSanitised    string
	preErr          error
	postVerdict     domain.ArmorVerdict
	postSanitised   string
	postErr         error
	preCalls        int
	postCalls       int
	lastPreTemplate string
	lastPrePrompt   string

	// prePrompts records EVERY PRE screen, in order. The G1'-3 contents leg
	// adds a second PRE call per turn, and lastPrePrompt alone cannot show
	// which text each call actually saw.
	prePrompts []string

	// preBlockOnceSubstring / preErrOnSubstring drive per-call behaviour so a
	// test can allow the flat prompt and block (or fault) only on the derived
	// contents text. Empty means "no per-call override", preserving the
	// existing preVerdict / preErr semantics exactly.
	preBlockOnceSubstring string
	preErrOnSubstring     string

	// postPrompts records EVERY POST screen, in order. The G1'-2 tool-call leg
	// adds a second POST call on a turn carrying both text and tool calls, and
	// a bare postCalls count cannot show which text each call actually saw.
	postPrompts []string

	// postBlockOnceSubstring / postErrOnSubstring are the POST-leg twins of the
	// two above, so a test can allow the completion and block (or fault) only
	// on the derived tool-call text.
	postBlockOnceSubstring string
	postErrOnSubstring     string
}

func (f *fakeArmor) SanitizeUserPrompt(ctx context.Context, template, prompt string) (domain.ArmorVerdict, string, error) {
	f.preCalls++
	f.lastPreTemplate = template
	f.lastPrePrompt = prompt
	f.prePrompts = append(f.prePrompts, prompt)
	if f.preErrOnSubstring != "" && strings.Contains(prompt, f.preErrOnSubstring) {
		return 0, "", errors.New("model armor unavailable")
	}
	if f.preBlockOnceSubstring != "" && strings.Contains(prompt, f.preBlockOnceSubstring) {
		return domain.ArmorVerdictBlock, "", nil
	}
	if f.preErr != nil {
		return 0, "", f.preErr
	}
	out := f.preSanitised
	if out == "" {
		out = prompt
	}
	return f.preVerdict, out, nil
}

func (f *fakeArmor) SanitizeModelResponse(ctx context.Context, template, response string) (domain.ArmorVerdict, string, error) {
	f.postCalls++
	f.postPrompts = append(f.postPrompts, response)
	if f.postErrOnSubstring != "" && strings.Contains(response, f.postErrOnSubstring) {
		return 0, "", errors.New("model armor unavailable")
	}
	if f.postBlockOnceSubstring != "" && strings.Contains(response, f.postBlockOnceSubstring) {
		return domain.ArmorVerdictBlock, "", nil
	}
	if f.postErr != nil {
		return 0, "", f.postErr
	}
	out := f.postSanitised
	if out == "" {
		out = response
	}
	return f.postVerdict, out, nil
}

type fakeBudget struct {
	state      *domain.BudgetState
	getErr     error
	debits     int64
	debitErr   error
	debitCalls int
}

func (f *fakeBudget) GetTenantBudget(ctx context.Context, tenantID string) (*domain.BudgetState, error) {
	if f.getErr != nil {
		return nil, f.getErr
	}
	return f.state, nil
}

func (f *fakeBudget) DebitSpent(ctx context.Context, tenantID string, usdMicrosDelta int64) error {
	f.debitCalls++
	if f.debitErr != nil {
		return f.debitErr
	}
	f.debits += usdMicrosDelta
	return nil
}

type fakeOutbox struct {
	events []domain.TokenUsageEvent
	err    error
}

func (f *fakeOutbox) EnqueueTokenUsageRecorded(ctx context.Context, evt domain.TokenUsageEvent) error {
	if f.err != nil {
		return f.err
	}
	f.events = append(f.events, evt)
	return nil
}

type fakePolicy struct {
	policy domain.AgentPolicy
	err    error

	// Captured args from the last ResolveAgentPolicy call — lets tests assert
	// the service forwards the agent-declared fallback chain (CR qgen 2026-06-01).
	gotRequestedModel domain.LogicalModelID
	gotFallbackModels []domain.LogicalModelID
}

func (f *fakePolicy) ResolveAgentPolicy(ctx context.Context, agentID, crewKind string, requestedModel domain.LogicalModelID, fallbackModels []domain.LogicalModelID) (domain.AgentPolicy, error) {
	f.gotRequestedModel = requestedModel
	f.gotFallbackModels = fallbackModels
	if f.err != nil {
		return domain.AgentPolicy{}, f.err
	}
	return f.policy, nil
}

// ----------------------------------------------------------------------------
// Helpers — fixed-time + deterministic-id builders so assertions are stable.
// ----------------------------------------------------------------------------

func fixedTime() time.Time { return time.Date(2026, 5, 24, 10, 0, 0, 0, time.UTC) }

func newTestService(t *testing.T, mods ...func(cfg *domain.ServiceConfig)) (*domain.Service, *fakeVendor, *fakeArmor, *fakeBudget, *fakeOutbox, *fakePolicy) {
	t.Helper()

	vendor := &fakeVendor{
		family: domain.VendorFamilyVertexGemini,
		resp: domain.VendorResponse{
			Completion: "completion-stub",
			Usage: domain.TokenUsage{
				InputTokens:  100,
				OutputTokens: 50,
				CostMicros:   12_500,
			},
			ModelVersion: "gemini-2.5-pro",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	armor := &fakeArmor{
		preVerdict:  domain.ArmorVerdictAllow,
		postVerdict: domain.ArmorVerdictAllow,
	}
	budget := &fakeBudget{
		state: &domain.BudgetState{
			TenantID:        "tenant-1",
			BudgetUSDMicros: 1_000_000,
			SpentUSDMicros:  0,
			Policy:          domain.BudgetPolicyBlock,
		},
	}
	outbox := &fakeOutbox{}
	policies := &fakePolicy{
		policy: domain.AgentPolicy{
			AgentID:                "qgen_question",
			ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
			Vendor:                 domain.VendorFamilyVertexGemini,
			ArmorTemplate:          "projects/chora-489812/locations/asia-southeast1/templates/chora-guardrail-balanced-dev",
		},
	}

	cfg := domain.ServiceConfig{
		Vendors:        []domain.VendorClient{vendor},
		Armor:          armor,
		Budget:         budget,
		Outbox:         outbox,
		Policies:       policies,
		GatewayVersion: "chora-model-gateway:test",
		Now:            fixedTime,
		NewID:          func() string { return "test-invocation-id" },
	}
	for _, m := range mods {
		m(&cfg)
	}
	svc, err := domain.NewService(cfg)
	require.NoError(t, err)
	return svc, vendor, armor, budget, outbox, policies
}

func happyRequest() domain.InvokeRequest {
	return domain.InvokeRequest{
		InvocationID:   "",
		TenantID:       "tenant-1",
		GCID:           "gcid-1",
		AgentID:        "qgen_question",
		CrewKind:       "qgen",
		LogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
		Prompt:         "What is the Scrum cadence for Sprint Planning?",
		Traceparent:    "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
	}
}

// ----------------------------------------------------------------------------
// Agent-driven fallback chain (CR qgen 2026-06-01)
// ----------------------------------------------------------------------------

// The service forwards the caller's primary + agent-declared fallback chain to
// the policy loader verbatim, so the loader can build policy.FallbackChain
// against the resolved vendor. This is the wiring the AGENT-DRIVEN tiering
// contract depends on.
func TestInvoke_ForwardsAgentDeclaredFallbackModels(t *testing.T) {
	svc, _, _, _, _, policies := newTestService(t)

	req := happyRequest()
	req.LogicalModelID = domain.LogicalModelID("gemini-3.1-pro-preview")
	req.FallbackModelIDs = []domain.LogicalModelID{"gemini-2.5-pro"}

	_, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)

	assert.Equal(t, domain.LogicalModelID("gemini-3.1-pro-preview"), policies.gotRequestedModel)
	assert.Equal(t, []domain.LogicalModelID{"gemini-2.5-pro"}, policies.gotFallbackModels)
}

// ----------------------------------------------------------------------------
// Happy path
// ----------------------------------------------------------------------------

func TestInvoke_HappyPath(t *testing.T) {
	svc, vendor, armor, budget, outbox, policies := newTestService(t)

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)

	// Response shape — every field populated.
	assert.Equal(t, "test-invocation-id", resp.InvocationID)
	assert.Equal(t, "completion-stub", resp.Completion)
	assert.Equal(t, domain.ArmorVerdictAllow, resp.ArmorPre)
	assert.Equal(t, domain.ArmorVerdictAllow, resp.ArmorPost)
	assert.Equal(t, string(domain.VendorFamilyVertexGemini), resp.Vendor)
	assert.Equal(t, "gemini-2.5-pro", resp.ModelVersion)
	assert.Equal(t, []string{"vertex_ai_gemini:gemini-2.5-pro"}, resp.FallbackChain)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
	assert.Equal(t, "chora-model-gateway:test", resp.GatewayVersion)
	assert.Equal(t, int64(100), resp.Usage.InputTokens)
	assert.Equal(t, int64(50), resp.Usage.OutputTokens)
	assert.Equal(t, int64(12_500), resp.Usage.CostMicros)

	// Vendor was called exactly once with the sanitised prompt + policy-resolved model.
	assert.Equal(t, 1, vendor.calls)
	assert.Equal(t, domain.LogicalModelID("gemini-2.5-pro"), vendor.lastReq.LogicalModelID)
	assert.Equal(t, "What is the Scrum cadence for Sprint Planning?", vendor.lastReq.Prompt)
	assert.Equal(t, "tenant-1", vendor.lastReq.TenantID)

	// Armor PRE + POST both called once with the resolved template.
	assert.Equal(t, 1, armor.preCalls)
	assert.Equal(t, 1, armor.postCalls)
	assert.Equal(t, "projects/chora-489812/locations/asia-southeast1/templates/chora-guardrail-balanced-dev", armor.lastPreTemplate)

	// Budget was debited atomically with the vendor-reported cost.
	assert.Equal(t, 1, budget.debitCalls)
	assert.Equal(t, int64(12_500), budget.debits)

	// Outbox event emitted with ADR-163 additive fields populated.
	require.Len(t, outbox.events, 1)
	evt := outbox.events[0]
	assert.Equal(t, "test-invocation-id", evt.UsageID)
	assert.Equal(t, "tenant-1", evt.TenantID)
	assert.Equal(t, "gcid-1", evt.GCID)
	assert.Equal(t, "gemini-2.5-pro", evt.ModelID)
	assert.Equal(t, int64(100), evt.InputTokens)
	assert.Equal(t, int64(50), evt.OutputTokens)
	assert.Equal(t, int64(12_500), evt.CostMicros)
	assert.Equal(t, "vertex_ai_gemini", evt.Vendor)
	assert.Equal(t, []string{"vertex_ai_gemini:gemini-2.5-pro"}, evt.FallbackChain)
	assert.Equal(t, domain.ArmorVerdictAllow, evt.ArmorPre)
	assert.Equal(t, domain.ArmorVerdictAllow, evt.ArmorPost)
	assert.Equal(t, "chora-model-gateway:test", evt.GatewayVersion)
	assert.Equal(t, "qgen_question", evt.AgentRole)
	assert.Equal(t, fixedTime(), evt.RecordedAt)

	// Policy was called once.
	_ = policies
}

// ----------------------------------------------------------------------------
// Image modality (W8, CR 2026-06-01)
// ----------------------------------------------------------------------------

// TestInvoke_ImageModality_ThreadsRequestAndPropagatesBytes asserts that
//  1. req.ResponseModality flows into the VendorRequest the service builds,
//  2. the vendor's ImageBytes + ImageMIMEType propagate to InvokeResponse,
//  3. the run still debits budget + emits its outbox event (image responses
//     reuse the response-agnostic ledger/budget machinery unchanged).
func TestInvoke_ImageModality_ThreadsRequestAndPropagatesBytes(t *testing.T) {
	imgBytes := []byte{0x89, 0x50, 0x4e, 0x47} // PNG magic — opaque to the domain
	// Declare the image vendor locally so we can assert on the exact instance
	// the service dispatched to (the closure-overridden one, not the default
	// fixture returned by newTestService) — same pattern as the fallback tests.
	imageVendor := &fakeVendor{
		family: domain.VendorFamilyVertexGemini,
		resp: domain.VendorResponse{
			// Image-only response: no completion text.
			Completion:    "",
			ImageBytes:    imgBytes,
			ImageMIMEType: "image/png",
			Usage:         domain.TokenUsage{InputTokens: 12, OutputTokens: 0, CostMicros: 15},
			ModelVersion:  "gemini-3-pro-image@001",
			FinishReason:  domain.FinishReasonComplete,
		},
	}
	svc, _, _, budget, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Vendors = []domain.VendorClient{imageVendor}
	})

	req := happyRequest()
	req.LogicalModelID = domain.LogicalModelID("gemini-3-pro-image")
	req.ResponseModality = "IMAGE"

	resp, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)

	// 1. Modality threaded into the vendor request verbatim.
	assert.Equal(t, "IMAGE", imageVendor.lastReq.ResponseModality)

	// 2. Image bytes + MIME propagate to the response; completion empty.
	assert.Equal(t, imgBytes, resp.ImageBytes)
	assert.Equal(t, "image/png", resp.ImageMIMEType)
	assert.Equal(t, "", resp.Completion)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)

	// 3. Response-agnostic machinery still ran: budget debited + outbox emitted.
	assert.Equal(t, 1, budget.debitCalls)
	assert.Equal(t, int64(15), budget.debits)
	require.Len(t, outbox.events, 1)
	assert.Equal(t, int64(12), outbox.events[0].InputTokens)
}

// TestInvoke_ImageModality_ArmorPostSkippedPassThrough asserts that for an
// image-ONLY response (no completion text), Armor POST is NOT called — there
// is no text to sanitise. Armor PRE on the prompt is still invoked, and the
// run still emits its outbox/budget. The POST verdict is reported as Bypassed
// (the pass-through marker), distinguishing "skipped because no text" from
// "Armor allowed the text".
func TestInvoke_ImageModality_ArmorPostSkippedPassThrough(t *testing.T) {
	svc, _, armor, budget, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Vendors = []domain.VendorClient{
			&fakeVendor{
				family: domain.VendorFamilyVertexGemini,
				resp: domain.VendorResponse{
					Completion:    "", // image-only
					ImageBytes:    []byte{0x89, 0x50, 0x4e, 0x47},
					ImageMIMEType: "image/png",
					Usage:         domain.TokenUsage{InputTokens: 12, OutputTokens: 0, CostMicros: 15},
					ModelVersion:  "gemini-3-pro-image@001",
					FinishReason:  domain.FinishReasonComplete,
				},
			},
		}
	})

	req := happyRequest()
	req.LogicalModelID = domain.LogicalModelID("gemini-3-pro-image")
	req.ResponseModality = "IMAGE"

	resp, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)

	// Armor PRE invoked on the prompt; POST skipped (no completion text to screen).
	assert.Equal(t, 1, armor.preCalls)
	assert.Equal(t, 0, armor.postCalls, "Armor POST must be skipped for image-only responses")
	assert.Equal(t, domain.ArmorVerdictAllow, resp.ArmorPre)
	assert.Equal(t, domain.ArmorVerdictBypassed, resp.ArmorPost, "image-only POST reports Bypassed pass-through")

	// Run still completes + emits ledger/budget.
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
	assert.Equal(t, 1, budget.debitCalls)
	require.Len(t, outbox.events, 1)
	assert.Equal(t, domain.ArmorVerdictBypassed, outbox.events[0].ArmorPost)
}

// TestInvoke_ImageModality_TextAlongsideImageStillScreened asserts that when
// the vendor returns BOTH text and an image, Armor POST still screens the
// text (image bytes are out of scope for the text guardrail, but accompanying
// text is NOT exempt).
func TestInvoke_ImageModality_TextAlongsideImageStillScreened(t *testing.T) {
	// Capture the armor instance locally so postCalls reflects the override.
	imageArmor := &fakeArmor{preVerdict: domain.ArmorVerdictAllow, postVerdict: domain.ArmorVerdictAllow}
	svc, _, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Vendors = []domain.VendorClient{
			&fakeVendor{
				family: domain.VendorFamilyVertexGemini,
				resp: domain.VendorResponse{
					Completion:    "here is your square:",
					ImageBytes:    []byte{0x89, 0x50, 0x4e, 0x47},
					ImageMIMEType: "image/png",
					Usage:         domain.TokenUsage{InputTokens: 12, OutputTokens: 4, CostMicros: 20},
					ModelVersion:  "gemini-3-pro-image@001",
					FinishReason:  domain.FinishReasonComplete,
				},
			},
		}
		cfg.Armor = imageArmor
	})

	req := happyRequest()
	req.LogicalModelID = domain.LogicalModelID("gemini-3-pro-image")
	req.ResponseModality = "IMAGE"

	resp, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)

	// Text present → Armor POST IS called; verdict reported.
	assert.Equal(t, 1, imageArmor.postCalls, "Armor POST must screen accompanying text")
	assert.Equal(t, domain.ArmorVerdictAllow, resp.ArmorPost)
	assert.Equal(t, "here is your square:", resp.Completion)
	assert.Equal(t, []byte{0x89, 0x50, 0x4e, 0x47}, resp.ImageBytes)
}

// TestInvoke_ImageModality_ArmorPostBlocksAccompanyingText asserts that when
// an image arrives WITH text and Armor POST blocks the text, the gateway
// short-circuits with a Model-Armor block — the image is NOT leaked past a
// text-block verdict.
func TestInvoke_ImageModality_ArmorPostBlocksAccompanyingText(t *testing.T) {
	svc, _, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Vendors = []domain.VendorClient{
			&fakeVendor{
				family: domain.VendorFamilyVertexGemini,
				resp: domain.VendorResponse{
					Completion:    "leaked secret here",
					ImageBytes:    []byte{0x89, 0x50, 0x4e, 0x47},
					ImageMIMEType: "image/png",
					Usage:         domain.TokenUsage{InputTokens: 12, OutputTokens: 4, CostMicros: 20},
					ModelVersion:  "gemini-3-pro-image@001",
					FinishReason:  domain.FinishReasonComplete,
				},
			},
		}
		cfg.Armor = &fakeArmor{preVerdict: domain.ArmorVerdictAllow, postVerdict: domain.ArmorVerdictBlock}
	})

	req := happyRequest()
	req.LogicalModelID = domain.LogicalModelID("gemini-3-pro-image")
	req.ResponseModality = "IMAGE"

	resp, err := svc.Invoke(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
	assert.Equal(t, domain.ArmorVerdictBlock, resp.ArmorPost)
	// Block path does NOT leak the image bytes.
	assert.Nil(t, resp.ImageBytes)
	assert.Equal(t, "", resp.ImageMIMEType)
	// Full-cost outbox event still emitted (vendor billed).
	require.Len(t, outbox.events, 1)
}

// ----------------------------------------------------------------------------
// Budget block — short-circuit before Armor PRE
// ----------------------------------------------------------------------------

func TestInvoke_BudgetExhausted_Block(t *testing.T) {
	svc, vendor, armor, budget, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		// Replace the default budget with an exhausted-block one.
		cfg.Budget = &fakeBudget{
			state: &domain.BudgetState{
				TenantID:        "tenant-1",
				BudgetUSDMicros: 100,
				SpentUSDMicros:  100, // exhausted
				Policy:          domain.BudgetPolicyBlock,
			},
		}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err) // budget-block is an in-band response, not an error
	assert.Equal(t, domain.FinishReasonBudgetBlock, resp.FinishReason)
	assert.Equal(t, "test-invocation-id", resp.InvocationID)
	assert.Equal(t, "", resp.Completion)

	// Vendor + Armor + Outbox NEVER called when budget blocks.
	assert.Equal(t, 0, vendor.calls)
	assert.Equal(t, 0, armor.preCalls)
	assert.Equal(t, 0, armor.postCalls)
	assert.Empty(t, outbox.events)
	_ = budget
}

// ----------------------------------------------------------------------------
// Budget downgrade — switch model + proceed
// ----------------------------------------------------------------------------

func TestInvoke_BudgetExhausted_Downgrade(t *testing.T) {
	svc, vendor, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Budget = &fakeBudget{
			state: &domain.BudgetState{
				TenantID:         "tenant-1",
				BudgetUSDMicros:  100,
				SpentUSDMicros:   100,
				Policy:           domain.BudgetPolicyDowngrade,
				DowngradeToModel: domain.LogicalModelID("gemini-2.5-flash-lite"),
			},
		}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
	// Vendor was called but with the downgrade model.
	assert.Equal(t, domain.LogicalModelID("gemini-2.5-flash-lite"), vendor.lastReq.LogicalModelID)
}

// ----------------------------------------------------------------------------
// Armor PRE block — short-circuit before vendor dispatch
// ----------------------------------------------------------------------------

func TestInvoke_ArmorPreBlock(t *testing.T) {
	svc, vendor, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Armor = &fakeArmor{preVerdict: domain.ArmorVerdictBlock}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
	assert.Equal(t, domain.ArmorVerdictBlock, resp.ArmorPre)
	assert.Equal(t, 0, vendor.calls)
	// Partial-cost outbox event IS emitted (input-only).
	require.Len(t, outbox.events, 1)
	assert.Equal(t, domain.ArmorVerdictBlock, outbox.events[0].ArmorPre)
	assert.Equal(t, int64(0), outbox.events[0].OutputTokens)
}

// ----------------------------------------------------------------------------
// Armor POST block — vendor was called but response is redacted
// ----------------------------------------------------------------------------

func TestInvoke_ArmorPostBlock(t *testing.T) {
	svc, vendor, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Armor = &fakeArmor{
			preVerdict:  domain.ArmorVerdictAllow,
			postVerdict: domain.ArmorVerdictBlock,
		}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
	assert.Equal(t, domain.ArmorVerdictBlock, resp.ArmorPost)
	assert.Equal(t, 1, vendor.calls)
	// Full-cost outbox event emitted (vendor billed for output).
	require.Len(t, outbox.events, 1)
	assert.Equal(t, int64(50), outbox.events[0].OutputTokens)
}

// ----------------------------------------------------------------------------
// Vendor error — surfaces as InvokeError(FinishReasonVendorError)
// ----------------------------------------------------------------------------

func TestInvoke_VendorError(t *testing.T) {
	svc, _, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		// Override the vendor in the registry.
		cfg.Vendors = []domain.VendorClient{
			&fakeVendor{family: domain.VendorFamilyVertexGemini, err: errors.New("vertex 503")},
		}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.Error(t, err)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Equal(t, domain.FinishReasonVendorError, ierr.Reason)
	// On error the response is zero-value — no leaked completion.
	assert.Empty(t, resp.Completion)
	assert.Empty(t, outbox.events)

	// Error message + Unwrap chain are useful for the gRPC adapter's
	// status-code mapping (Phase 2.4). Exercise them so coverage proves
	// the InvokeError API surface is intact.
	assert.Contains(t, ierr.Error(), "vendor_error")
	assert.Contains(t, ierr.Error(), "vertex 503")
	assert.ErrorIs(t, errors.Unwrap(ierr), ierr.Inner)
}

// ----------------------------------------------------------------------------
// Policy resolve failure — surfaces as InvokeError too
// ----------------------------------------------------------------------------

func TestInvoke_PolicyResolveError(t *testing.T) {
	svc, _, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Policies = &fakePolicy{err: errors.New("policy yaml not loaded")}
	})

	_, err := svc.Invoke(context.Background(), happyRequest())
	require.Error(t, err)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Equal(t, "policy resolve failed", ierr.Detail)
	assert.Empty(t, outbox.events)
}

// ----------------------------------------------------------------------------
// Budget GetTenantBudget error — surfaces as InvokeError
// ----------------------------------------------------------------------------

func TestInvoke_BudgetGetError(t *testing.T) {
	svc, _, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Budget = &fakeBudget{getErr: errors.New("pg timeout")}
	})

	_, err := svc.Invoke(context.Background(), happyRequest())
	require.Error(t, err)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Equal(t, "budget repo unavailable", ierr.Detail)
	assert.Empty(t, outbox.events)
}

// ----------------------------------------------------------------------------
// Vendor family registry — vendor present but agent policy specifies an
// unregistered family
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// Fallback chain — primary fails, fallback #1 succeeds
// ----------------------------------------------------------------------------

func TestInvoke_FallbackChain_SecondAttemptSucceeds(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyVertexGemini, err: errors.New("vertex 503")}
	fallback := &fakeVendor{
		family: domain.VendorFamilyOpenAI,
		resp: domain.VendorResponse{
			Completion:   "fallback-completion",
			Usage:        domain.TokenUsage{InputTokens: 100, OutputTokens: 40, CostMicros: 6_000},
			ModelVersion: "gpt-4o-mini",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	svc, _, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Vendors = []domain.VendorClient{primary, fallback}
		cfg.Policies = &fakePolicy{
			policy: domain.AgentPolicy{
				AgentID:                "qgen_question",
				ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
				Vendor:                 domain.VendorFamilyVertexGemini,
				FallbackChain: []domain.AgentPolicyFallback{
					{Vendor: domain.VendorFamilyOpenAI, ResolvedLogicalModelID: domain.LogicalModelID("gpt-4o-mini")},
				},
				ArmorTemplate: "projects/chora-489812/locations/asia-southeast1/templates/chora-guardrail-balanced-dev",
			},
		}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, "fallback-completion", resp.Completion)
	assert.Equal(t, 1, primary.calls)
	assert.Equal(t, 1, fallback.calls)
	// The fallback_chain on the response records BOTH attempts in order.
	assert.Equal(t, []string{"vertex_ai_gemini:gemini-2.5-pro", "openai_byoa:gpt-4o-mini"}, resp.FallbackChain)
	// Resolved vendor + model reflect the SUCCEEDING attempt, not the primary.
	assert.Equal(t, string(domain.VendorFamilyOpenAI), resp.Vendor)
	assert.Equal(t, "gpt-4o-mini", resp.ModelVersion)
	require.Len(t, outbox.events, 1)
	assert.Equal(t, "openai_byoa", outbox.events[0].Vendor)
}

func TestInvoke_FallbackChain_AllExhausted(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyVertexGemini, err: errors.New("vertex 503")}
	fallback := &fakeVendor{family: domain.VendorFamilyOpenAI, err: errors.New("openai 429")}
	svc, _, _, _, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Vendors = []domain.VendorClient{primary, fallback}
		cfg.Policies = &fakePolicy{
			policy: domain.AgentPolicy{
				AgentID:                "qgen_question",
				ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
				Vendor:                 domain.VendorFamilyVertexGemini,
				FallbackChain: []domain.AgentPolicyFallback{
					{Vendor: domain.VendorFamilyOpenAI, ResolvedLogicalModelID: domain.LogicalModelID("gpt-4o-mini")},
				},
			},
		}
	})

	_, err := svc.Invoke(context.Background(), happyRequest())
	require.Error(t, err)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Equal(t, domain.FinishReasonVendorError, ierr.Reason)
	assert.Contains(t, ierr.Error(), "all 2 provider attempts failed")
	assert.Equal(t, 1, primary.calls)
	assert.Equal(t, 1, fallback.calls)
	assert.Empty(t, outbox.events) // no successful dispatch = no outbox emit
}

func TestInvoke_FallbackChain_UnregisteredVendorSkipped(t *testing.T) {
	// Primary vendor is registered + failing; fallback references a
	// vendor family that's NOT in the registry. Service must skip the
	// unregistered fallback (rather than blow up) + then fail-loud since
	// all real attempts failed.
	primary := &fakeVendor{family: domain.VendorFamilyVertexGemini, err: errors.New("vertex 503")}
	svc, _, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Vendors = []domain.VendorClient{primary}
		cfg.Policies = &fakePolicy{
			policy: domain.AgentPolicy{
				AgentID:                "qgen_question",
				ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
				Vendor:                 domain.VendorFamilyVertexGemini,
				FallbackChain: []domain.AgentPolicyFallback{
					{Vendor: domain.VendorFamilyAnthropic, ResolvedLogicalModelID: domain.LogicalModelID("claude-opus-4-7")},
				},
			},
		}
	})

	_, err := svc.Invoke(context.Background(), happyRequest())
	require.Error(t, err)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Error(), "all 2 provider attempts failed")
}

func TestInvoke_UnregisteredVendorFamily(t *testing.T) {
	svc, _, _, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Policies = &fakePolicy{
			policy: domain.AgentPolicy{
				AgentID:                "qgen_question",
				ResolvedLogicalModelID: domain.LogicalModelID("claude-opus-4-7"),
				Vendor:                 domain.VendorFamilyAnthropic, // not registered in test fixture
			},
		}
	})

	_, err := svc.Invoke(context.Background(), happyRequest())
	require.Error(t, err)
	var ierr *domain.InvokeError
	require.True(t, errors.As(err, &ierr))
	assert.Contains(t, ierr.Error(), "anthropic_byoa")
}

// ----------------------------------------------------------------------------
// Budget alert mode — proceeds AND debits (the alert is an observability
// signal, not a debit exemption; matches the Python reference where
// alert-mode tenants are billed but flagged for the O+ dashboard).
// ----------------------------------------------------------------------------

func TestInvoke_BudgetExhausted_Alert(t *testing.T) {
	svc, vendor, _, budget, outbox, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		// Modify the existing budget in place so the returned handle
		// observes the debit.
		b := cfg.Budget.(*fakeBudget)
		b.state = &domain.BudgetState{
			TenantID:        "tenant-1",
			BudgetUSDMicros: 100,
			SpentUSDMicros:  100,
			Policy:          domain.BudgetPolicyAlert,
		}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
	assert.Equal(t, 1, vendor.calls)
	// Alert mode DOES debit (matching the Python reference) + emits the
	// outbox event (cost visibility for the O+ dashboard).
	assert.Equal(t, 1, budget.debitCalls)
	assert.Equal(t, int64(12_500), budget.debits)
	require.Len(t, outbox.events, 1)
}

// ----------------------------------------------------------------------------
// Nil budget repo state (tenant has no configured budget) — proceeds as Allow
// ----------------------------------------------------------------------------

func TestInvoke_NoConfiguredBudget(t *testing.T) {
	svc, vendor, _, budget, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Budget = &fakeBudget{state: nil}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, domain.FinishReasonComplete, resp.FinishReason)
	assert.Equal(t, 1, vendor.calls)
	assert.Equal(t, 0, budget.debitCalls) // no budget configured → no debit
}

// ----------------------------------------------------------------------------
// Permissive tier (no Armor template) — bypasses both PRE + POST
// ----------------------------------------------------------------------------

func TestInvoke_PermissiveTier_NoArmor(t *testing.T) {
	svc, _, armor, _, _, _ := newTestService(t, func(cfg *domain.ServiceConfig) {
		cfg.Policies = &fakePolicy{
			policy: domain.AgentPolicy{
				AgentID:                "qgen_question",
				ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
				Vendor:                 domain.VendorFamilyVertexGemini,
				ArmorTemplate:          "", // permissive tier
			},
		}
	})

	resp, err := svc.Invoke(context.Background(), happyRequest())
	require.NoError(t, err)
	assert.Equal(t, domain.ArmorVerdictBypassed, resp.ArmorPre)
	assert.Equal(t, domain.ArmorVerdictBypassed, resp.ArmorPost)
	assert.Equal(t, 0, armor.preCalls)
	assert.Equal(t, 0, armor.postCalls)
}

// ----------------------------------------------------------------------------
// FinishReason.String — pure-function test for OTLP span attribute output
// ----------------------------------------------------------------------------

func TestFinishReason_String(t *testing.T) {
	cases := map[domain.FinishReason]string{
		domain.FinishReasonComplete:        "complete",
		domain.FinishReasonMaxTokens:       "max_tokens",
		domain.FinishReasonModelArmorBlock: "model_armor_block",
		domain.FinishReasonBudgetBlock:     "budget_block",
		domain.FinishReasonVendorError:     "vendor_error",
		domain.FinishReasonUnspecified:     "unspecified",
		domain.FinishReason(999):           "unspecified",
	}
	for fr, want := range cases {
		assert.Equal(t, want, fr.String())
	}
}

// ----------------------------------------------------------------------------
// ArmorVerdict.String — same shape as above
// ----------------------------------------------------------------------------

func TestArmorVerdict_String(t *testing.T) {
	cases := map[domain.ArmorVerdict]string{
		domain.ArmorVerdictAllow:       "allow",
		domain.ArmorVerdictBlock:       "block",
		domain.ArmorVerdictSanitise:    "sanitise",
		domain.ArmorVerdictError:       "error",
		domain.ArmorVerdictBypassed:    "bypassed",
		domain.ArmorVerdictUnspecified: "unspecified",
		domain.ArmorVerdict(999):       "unspecified",
	}
	for v, want := range cases {
		assert.Equal(t, want, v.String())
	}
}

// ----------------------------------------------------------------------------
// InvokeError.Error — covers both Inner-set and Inner-nil paths
// ----------------------------------------------------------------------------

func TestInvokeError_Error(t *testing.T) {
	withInner := &domain.InvokeError{
		Reason: domain.FinishReasonVendorError,
		Detail: "vertex 503",
		Inner:  errors.New("upstream timeout"),
	}
	assert.Contains(t, withInner.Error(), "vertex 503")
	assert.Contains(t, withInner.Error(), "upstream timeout")

	noInner := &domain.InvokeError{
		Reason: domain.FinishReasonBudgetBlock,
		Detail: "tenant budget exhausted",
	}
	assert.Contains(t, noInner.Error(), "tenant budget exhausted")
	assert.NotContains(t, noInner.Error(), "<nil>")
}

// ----------------------------------------------------------------------------
// NewService — nil-vendor-element + empty-family branches
// ----------------------------------------------------------------------------

func TestNewService_NilVendorElement(t *testing.T) {
	cfg := baseServiceConfig()
	cfg.Vendors = []domain.VendorClient{nil}
	_, err := domain.NewService(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil VendorClient")
}

func TestNewService_EmptyVendorFamily(t *testing.T) {
	cfg := baseServiceConfig()
	cfg.Vendors = []domain.VendorClient{&fakeVendor{family: ""}}
	_, err := domain.NewService(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Family() returned empty")
}

// ----------------------------------------------------------------------------
// NewService — invariants on the constructor
// ----------------------------------------------------------------------------

func TestNewService_RequiresAllPorts(t *testing.T) {
	// Each subtest omits exactly one port to verify the constructor's
	// fail-loud guards (feedback_no_stubs_real_wiring).
	cases := []struct {
		name string
		mut  func(c *domain.ServiceConfig)
		want string
	}{
		{"missing armor", func(c *domain.ServiceConfig) { c.Armor = nil }, "ArmorClient required"},
		{"missing budget", func(c *domain.ServiceConfig) { c.Budget = nil }, "BudgetRepo required"},
		{"missing outbox", func(c *domain.ServiceConfig) { c.Outbox = nil }, "OutboxWriter required"},
		{"missing policies", func(c *domain.ServiceConfig) { c.Policies = nil }, "PolicyLoader required"},
		{"empty vendors", func(c *domain.ServiceConfig) { c.Vendors = nil }, "VendorClient required"},
		{"empty gateway version", func(c *domain.ServiceConfig) { c.GatewayVersion = "" }, "GatewayVersion required"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := baseServiceConfig()
			tc.mut(&cfg)
			_, err := domain.NewService(cfg)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.want)
		})
	}
}

func TestNewService_RejectsDuplicateVendor(t *testing.T) {
	cfg := baseServiceConfig()
	cfg.Vendors = []domain.VendorClient{
		&fakeVendor{family: domain.VendorFamilyVertexGemini},
		&fakeVendor{family: domain.VendorFamilyVertexGemini},
	}
	_, err := domain.NewService(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate VendorClient")
}

func baseServiceConfig() domain.ServiceConfig {
	return domain.ServiceConfig{
		Vendors:        []domain.VendorClient{&fakeVendor{family: domain.VendorFamilyVertexGemini}},
		Armor:          &fakeArmor{},
		Budget:         &fakeBudget{},
		Outbox:         &fakeOutbox{},
		Policies:       &fakePolicy{},
		GatewayVersion: "test",
		Now:            fixedTime,
		NewID:          func() string { return "id" },
	}
}

// ----------------------------------------------------------------------------
// BudgetState.Decide — pure-function test
// ----------------------------------------------------------------------------

func TestBudgetState_Decide(t *testing.T) {
	cases := []struct {
		name  string
		state domain.BudgetState
		want  domain.BudgetDecision
	}{
		{"under budget", domain.BudgetState{BudgetUSDMicros: 100, SpentUSDMicros: 50, Policy: domain.BudgetPolicyBlock}, domain.BudgetAllow},
		{"at budget — block", domain.BudgetState{BudgetUSDMicros: 100, SpentUSDMicros: 100, Policy: domain.BudgetPolicyBlock}, domain.BudgetBlock},
		{"over budget — downgrade", domain.BudgetState{BudgetUSDMicros: 100, SpentUSDMicros: 200, Policy: domain.BudgetPolicyDowngrade}, domain.BudgetDowngrade},
		{"over budget — alert", domain.BudgetState{BudgetUSDMicros: 100, SpentUSDMicros: 200, Policy: domain.BudgetPolicyAlert}, domain.BudgetAlert},
		{"unknown policy defaults to block", domain.BudgetState{BudgetUSDMicros: 100, SpentUSDMicros: 100, Policy: "weird"}, domain.BudgetBlock},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, tc.state.Decide())
		})
	}
}

// ----------------------------------------------------------------------------
// ArmorVerdict.IsTerminalBlock — pure-function test
// ----------------------------------------------------------------------------

func TestArmorVerdict_IsTerminalBlock(t *testing.T) {
	assert.True(t, domain.ArmorVerdictBlock.IsTerminalBlock())
	assert.True(t, domain.ArmorVerdictError.IsTerminalBlock())
	assert.False(t, domain.ArmorVerdictAllow.IsTerminalBlock())
	assert.False(t, domain.ArmorVerdictSanitise.IsTerminalBlock())
	assert.False(t, domain.ArmorVerdictBypassed.IsTerminalBlock())
}
