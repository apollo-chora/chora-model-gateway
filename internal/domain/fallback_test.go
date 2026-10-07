package domain_test

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ----------------------------------------------------------------------------
// Fakes for the fallback engine
// ----------------------------------------------------------------------------

// statusError builds a provider error carrying an HTTP status, the shape every
// vendor adapter returns for a non-2xx response.
func statusError(status int) error {
	return &domain.ProviderError{Status: status, Endpoint: "https://provider.example/v1", Body: "upstream said no"}
}

// scriptedProvider returns a scripted sequence of (response, error) results,
// one per call, so a test can drive a partial-usage-then-fail sequence.
type scriptedProvider struct {
	family  domain.VendorFamily
	results []scriptedResult
	calls   int
}

type scriptedResult struct {
	resp domain.VendorResponse
	err  error
}

func (p *scriptedProvider) Family() domain.VendorFamily { return p.family }

func (p *scriptedProvider) Generate(_ context.Context, _ domain.VendorRequest) (domain.VendorResponse, error) {
	i := p.calls
	p.calls++
	if i >= len(p.results) {
		i = len(p.results) - 1
	}
	return p.results[i].resp, p.results[i].err
}

// mapModelResolver resolves a logical model id to its registry metadata, so a
// test can give different targets different capabilities and credentials.
type mapModelResolver struct {
	infos map[domain.LogicalModelID]domain.ModelInfo
}

func (m *mapModelResolver) Resolve(_ context.Context, id domain.LogicalModelID) (domain.ModelInfo, error) {
	info, ok := m.infos[id]
	if !ok {
		return domain.ModelInfo{}, errors.New("model is not in the registry")
	}
	return info, nil
}

func newTestEngine(t *testing.T, providers []domain.Provider, mods ...func(*domain.FallbackConfig)) *domain.FallbackEngine {
	t.Helper()
	cfg := domain.FallbackConfig{
		Providers: providers,
		// A no-op sleeper so a same-provider backoff never slows the suite.
		Sleep: func(context.Context, time.Duration) error { return nil },
	}
	for _, m := range mods {
		m(&cfg)
	}
	engine, err := domain.NewFallbackEngine(cfg)
	require.NoError(t, err)
	return engine
}

func baseRun(targets ...domain.FallbackTarget) domain.FallbackRun {
	return domain.FallbackRun{
		Targets:            targets,
		Base:               domain.VendorRequest{Prompt: "hello"},
		RequiredCapability: "chat",
	}
}

func openaiTarget() domain.FallbackTarget {
	return domain.FallbackTarget{Vendor: domain.VendorFamilyOpenAI, LogicalModelID: domain.LogicalModelID("gpt-4o")}
}

func anthropicTarget() domain.FallbackTarget {
	return domain.FallbackTarget{Vendor: domain.VendorFamilyAnthropic, LogicalModelID: domain.LogicalModelID("claude-sonnet-4-5")}
}

// ----------------------------------------------------------------------------
// The retry matrix is the spec, written down
// ----------------------------------------------------------------------------

// Every row of the ratified table is asserted literally. If someone changes a
// decision, this test fails and the change has to be argued for.
func TestDefaultRetryMatrix_ExplicitTable(t *testing.T) {
	matrix := domain.DefaultRetryMatrix()

	cases := []struct {
		class        domain.FailureClass
		sameProvider domain.RetryDecision
		fallback     domain.RetryDecision
	}{
		{domain.FailureTimeout, domain.RetryMaybe, domain.RetryYes},
		{domain.FailureRateLimited, domain.RetryMaybe, domain.RetryYes},
		{domain.FailureProvider5xx, domain.RetryMaybe, domain.RetryYes},
		{domain.FailureInvalidCredentials, domain.RetryNo, domain.RetryMaybe},
		{domain.FailureMalformedRequest, domain.RetryNo, domain.RetryNo},
		{domain.FailureUnsupportedCapability, domain.RetryNo, domain.RetryMaybe},
		{domain.FailurePrePolicyRejection, domain.RetryNo, domain.RetryNo},
		{domain.FailurePostPolicyRejection, domain.RetryNo, domain.RetryNo},
		{domain.FailureBudgetExhausted, domain.RetryNo, domain.RetryNo},
		{domain.FailureLedgerFailure, domain.RetryNo, domain.RetryNo},
		{domain.FailureInternalInvariant, domain.RetryNo, domain.RetryNo},
	}

	for _, tc := range cases {
		t.Run(tc.class.String(), func(t *testing.T) {
			rule := matrix.Lookup(tc.class)
			assert.Equal(t, tc.class, rule.Class)
			assert.Equal(t, tc.sameProvider, rule.SameProvider, "same-provider retry decision")
			assert.Equal(t, tc.fallback, rule.Fallback, "fallback decision")
			assert.NotEmpty(t, rule.Rationale, "every row carries its rationale")
		})
	}

	// The table is complete: every ratified class, plus the fail-closed
	// FailureUnknown row that catches anything the gateway cannot explain.
	assert.Len(t, matrix.Rules(), len(cases)+1)
}

// A zero-value matrix has no rules and must fail closed rather than silently
// permitting anything.
func TestRetryMatrix_ZeroValueFailsClosed(t *testing.T) {
	var empty domain.RetryMatrix
	rule := empty.Lookup(domain.FailureTimeout)
	assert.Equal(t, domain.RetryNo, rule.SameProvider)
	assert.Equal(t, domain.RetryNo, rule.Fallback)
	assert.Empty(t, empty.Rules())
}

// A policy refusal is never an automatic fallback — the explicit safety rule
// this port was asked to encode.
func TestRetryMatrix_PolicyRejectionsNeverFallBack(t *testing.T) {
	matrix := domain.DefaultRetryMatrix()
	for _, class := range []domain.FailureClass{
		domain.FailurePrePolicyRejection,
		domain.FailurePostPolicyRejection,
	} {
		rule := matrix.Lookup(class)
		assert.Equal(t, domain.RetryNo, rule.Fallback, "%s must not fall back", class)
		assert.Equal(t, domain.RetryNo, rule.SameProvider, "%s must not retry", class)
	}
}

func TestRetryDecision_String(t *testing.T) {
	assert.Equal(t, "no", domain.RetryNo.String())
	assert.Equal(t, "maybe", domain.RetryMaybe.String())
	assert.Equal(t, "yes", domain.RetryYes.String())
}

// ----------------------------------------------------------------------------
// Classification
// ----------------------------------------------------------------------------

func TestClassifyError(t *testing.T) {
	cases := []struct {
		name  string
		err   error
		class domain.FailureClass
	}{
		{"nil", nil, domain.FailureUnknown},
		{"429", statusError(429), domain.FailureRateLimited},
		{"500", statusError(500), domain.FailureProvider5xx},
		{"503", statusError(503), domain.FailureProvider5xx},
		{"401", statusError(401), domain.FailureInvalidCredentials},
		{"403", statusError(403), domain.FailureInvalidCredentials},
		{"400", statusError(400), domain.FailureMalformedRequest},
		{"422", statusError(422), domain.FailureMalformedRequest},
		{"404", statusError(404), domain.FailureUnsupportedCapability},
		{"408", statusError(408), domain.FailureTimeout},
		{"deadline exceeded", context.DeadlineExceeded, domain.FailureTimeout},
		{"wrapped deadline exceeded", fmt.Errorf("transport: %w", context.DeadlineExceeded), domain.FailureTimeout},
		{"canceled", context.Canceled, domain.FailureUnknown},
		{"timeout net error", &timeoutError{}, domain.FailureTimeout},
		{"transport error with no status", errors.New("connection refused"), domain.FailureProvider5xx},
		{"capability", &domain.CapabilityError{Model: "text-model", Capability: "image"}, domain.FailureUnsupportedCapability},
		{"credential", &domain.CredentialError{Model: "m", APIKeyEnv: "K"}, domain.FailureInvalidCredentials},
		{"no provider", &domain.NoProviderError{Family: domain.VendorFamilyAnthropic}, domain.FailureProvider5xx},
		{"config", &domain.ConfigError{Detail: "model is not in the registry"}, domain.FailureUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.class, domain.ClassifyError(tc.err))
		})
	}
}

// A typed gateway error is classified by type even when it carries an HTTP
// status: a capability refusal that happens to be a 404 must not read as a
// malformed request.
func TestClassifyError_TypedErrorWinsOverStatus(t *testing.T) {
	err := &domain.ProviderError{Status: 400, Endpoint: "x", Body: "y"}
	require.Equal(t, domain.FailureMalformedRequest, domain.ClassifyError(err))
	assert.Equal(t, domain.FailureInvalidCredentials,
		domain.ClassifyError(&domain.CredentialError{Model: "m", APIKeyEnv: "K"}))
}

type timeoutError struct{}

func (e *timeoutError) Error() string   { return "i/o timeout" }
func (e *timeoutError) Timeout() bool   { return true }
func (e *timeoutError) Temporary() bool { return true }

func TestClassifyFinishReason(t *testing.T) {
	assert.Equal(t, domain.FailurePostPolicyRejection, domain.ClassifyFinishReason(domain.FinishReasonModelArmorBlock))
	assert.Equal(t, domain.FailureBudgetExhausted, domain.ClassifyFinishReason(domain.FinishReasonBudgetBlock))
	assert.Equal(t, domain.FailureProvider5xx, domain.ClassifyFinishReason(domain.FinishReasonVendorError))
	assert.Equal(t, domain.FailureUnknown, domain.ClassifyFinishReason(domain.FinishReasonComplete))
}

// ----------------------------------------------------------------------------
// The walk: what falls back and what does not
// ----------------------------------------------------------------------------

// A 429 on the primary falls back to the next provider, and only the winner's
// cost is charged. The primary is attempted exactly once: same-provider retry
// is opt-in, and the default is off.
func TestFallbackEngine_RateLimited_FallsBackAndChargesOnlyTheWinner(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyOpenAI, err: statusError(429)}
	fallback := &fakeVendor{
		family: domain.VendorFamilyAnthropic,
		resp: domain.VendorResponse{
			Completion:   "answered by anthropic",
			Usage:        domain.TokenUsage{InputTokens: 100, OutputTokens: 50, CostMicros: 2_500},
			ModelVersion: "claude-sonnet-4-5",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	engine := newTestEngine(t, []domain.Provider{primary, fallback})

	out, err := engine.Run(context.Background(), baseRun(openaiTarget(), anthropicTarget()))
	require.NoError(t, err)

	assert.Equal(t, 1, primary.calls, "429 does not retry the same provider by default")
	assert.Equal(t, 1, fallback.calls)
	assert.Equal(t, "answered by anthropic", out.Response.Completion)
	assert.Equal(t, []string{"openai_byoa:gpt-4o", "anthropic_byoa:claude-sonnet-4-5"}, out.Chain)

	require.Len(t, out.Attempts, 2)
	assert.False(t, out.Attempts[0].Billable, "the failed attempt is not billable")
	assert.Equal(t, domain.FailureRateLimited, out.Attempts[0].Class)
	assert.Equal(t, 429, *out.Attempts[0].UpstreamStatus)
	assert.True(t, out.Attempts[1].Billable)

	assert.Equal(t, int64(2_500), out.ProviderCharge.CostMicros,
		"only the winning provider's cost is charged")
	assert.Equal(t, string(domain.VendorFamilyAnthropic), out.ProviderCharge.Vendor)
}

// The accounting rule the port was asked for by name: a provider that produced
// tokens and THEN errored has not served the turn. Its partial output is
// recorded as evidence and contributes nothing to the charge.
func TestFallbackEngine_PartialOutputFromFailedAttemptIsNotBillable(t *testing.T) {
	primary := &scriptedProvider{
		family: domain.VendorFamilyOpenAI,
		results: []scriptedResult{{
			// The provider streamed 2000 output tokens before dying.
			resp: domain.VendorResponse{
				Completion: "half an ans",
				Usage:      domain.TokenUsage{InputTokens: 400, OutputTokens: 2_000, CostMicros: 5_000},
			},
			err: statusError(500),
		}},
	}
	fallback := &fakeVendor{
		family: domain.VendorFamilyAnthropic,
		resp: domain.VendorResponse{
			Completion:   "the whole answer",
			Usage:        domain.TokenUsage{InputTokens: 400, OutputTokens: 2_500, CostMicros: 3_000},
			ModelVersion: "claude-sonnet-4-5",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	engine := newTestEngine(t, []domain.Provider{primary, fallback})

	out, err := engine.Run(context.Background(), baseRun(openaiTarget(), anthropicTarget()))
	require.NoError(t, err)

	assert.Equal(t, int64(3_000), out.ProviderCharge.CostMicros,
		"only Anthropic's 2500-token success is charged, not OpenAI's 2000-token partial output")
	require.Len(t, out.Attempts, 2)
	assert.False(t, out.Attempts[0].Billable)
	assert.Equal(t, int64(2_000), out.Attempts[0].Usage.OutputTokens,
		"the partial usage is kept on the audit trail")
	assert.Equal(t, int64(5_000), out.Attempts[0].Usage.CostMicros,
		"the partial cost is recorded but never charged")
	assert.True(t, out.Attempts[1].Billable)
	assert.Equal(t, int64(2_500), out.Response.Usage.OutputTokens,
		"the response carries the winning attempt's usage only")
}

// A timeout falls back unconditionally.
func TestFallbackEngine_Timeout_FallsBack(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyOpenAI, err: context.DeadlineExceeded}
	fallback := &fakeVendor{
		family: domain.VendorFamilyAnthropic,
		resp: domain.VendorResponse{
			Completion:   "ok",
			Usage:        domain.TokenUsage{CostMicros: 10},
			ModelVersion: "claude-sonnet-4-5",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	engine := newTestEngine(t, []domain.Provider{primary, fallback})

	out, err := engine.Run(context.Background(), baseRun(openaiTarget(), anthropicTarget()))
	require.NoError(t, err)
	assert.Equal(t, "ok", out.Response.Completion)
	assert.Equal(t, domain.FailureTimeout, out.Attempts[0].Class)
	assert.False(t, out.Attempts[0].Billable)
}

// A malformed request is the caller's payload: every provider rejects it the
// same way, so the chain is not walked at all.
func TestFallbackEngine_MalformedRequest_DoesNotWalkTheChain(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyOpenAI, err: statusError(400)}
	fallback := &fakeVendor{family: domain.VendorFamilyAnthropic, resp: domain.VendorResponse{Completion: "ok"}}
	engine := newTestEngine(t, []domain.Provider{primary, fallback})

	_, err := engine.Run(context.Background(), baseRun(openaiTarget(), anthropicTarget()))
	require.Error(t, err)

	assert.Equal(t, 1, primary.calls)
	assert.Equal(t, 0, fallback.calls, "a malformed request must not burn the chain")

	var exhaustedErr *domain.FallbackExhaustedError
	require.ErrorAs(t, err, &exhaustedErr)
	assert.Equal(t, domain.FailureMalformedRequest, exhaustedErr.Last.Class)
	assert.Len(t, exhaustedErr.Attempts, 1)
}

// Same-provider retry is opt-in. When configured, a retryable failure is
// retried on the SAME provider before the walk advances.
func TestFallbackEngine_SameProviderRetry_WhenConfigured(t *testing.T) {
	primary := &scriptedProvider{
		family: domain.VendorFamilyOpenAI,
		results: []scriptedResult{
			{err: statusError(503)},
			{err: statusError(503)},
			{resp: domain.VendorResponse{
				Completion:   "recovered",
				Usage:        domain.TokenUsage{CostMicros: 700},
				ModelVersion: "gpt-4o",
				FinishReason: domain.FinishReasonComplete,
			}},
		},
	}
	fallback := &fakeVendor{family: domain.VendorFamilyAnthropic, resp: domain.VendorResponse{Completion: "fallback"}}
	engine := newTestEngine(t, []domain.Provider{primary, fallback}, func(cfg *domain.FallbackConfig) {
		cfg.MaxSameProviderRetries = 2
	})

	out, err := engine.Run(context.Background(), baseRun(openaiTarget(), anthropicTarget()))
	require.NoError(t, err)

	assert.Equal(t, 3, primary.calls, "two extra same-provider attempts")
	assert.Equal(t, 0, fallback.calls, "the fallback was never needed")
	assert.Equal(t, "recovered", out.Response.Completion)
	assert.Equal(t, 2, out.Attempts[0].SameProviderRetries)
}

// The same-provider retry budget does not override the matrix: a malformed
// request is never retried, however many retries are configured.
func TestFallbackEngine_SameProviderRetry_NotForMalformedRequest(t *testing.T) {
	primary := &scriptedProvider{
		family: domain.VendorFamilyOpenAI,
		results: []scriptedResult{
			{err: statusError(400)},
			{resp: domain.VendorResponse{Completion: "should never run"}},
		},
	}
	engine := newTestEngine(t, []domain.Provider{primary}, func(cfg *domain.FallbackConfig) {
		cfg.MaxSameProviderRetries = 3
	})

	_, err := engine.Run(context.Background(), baseRun(openaiTarget()))
	require.Error(t, err)
	assert.Equal(t, 1, primary.calls, "a 400 is not retried")
}

// Invalid credentials: the same key cannot start working, so the provider is
// not retried — but the registry may hold a target that can authenticate.
func TestFallbackEngine_InvalidCredentials_FallsBackWhenRegistryPermits(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyOpenAI}
	fallback := &fakeVendor{
		family: domain.VendorFamilyAnthropic,
		resp: domain.VendorResponse{
			Completion:   "authenticated elsewhere",
			Usage:        domain.TokenUsage{CostMicros: 20},
			ModelVersion: "claude-sonnet-4-5",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	models := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"gpt-4o":            {ID: "gpt-4o", APIKeyEnv: "OPENAI_API_KEY", APIKey: ""},
		"claude-sonnet-4-5": {ID: "claude-sonnet-4-5", APIKeyEnv: "ANTHROPIC_API_KEY", APIKey: "sk-live"},
	}}
	engine := newTestEngine(t, []domain.Provider{primary, fallback}, func(cfg *domain.FallbackConfig) {
		cfg.Models = models
	})

	out, err := engine.Run(context.Background(), baseRun(openaiTarget(), anthropicTarget()))
	require.NoError(t, err)
	assert.Equal(t, 0, primary.calls, "an empty credential is caught before dispatch")
	assert.Equal(t, 1, fallback.calls)
	assert.Equal(t, "authenticated elsewhere", out.Response.Completion)
	assert.Equal(t, domain.FailureInvalidCredentials, out.Attempts[0].Class)
}

// ...and when no remaining target can authenticate either, the walk stops
// rather than trying a target that will fail identically.
func TestFallbackEngine_InvalidCredentials_StopsWhenNoTargetCanAuthenticate(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyOpenAI}
	fallback := &fakeVendor{family: domain.VendorFamilyAnthropic}
	models := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"gpt-4o":            {ID: "gpt-4o", APIKeyEnv: "OPENAI_API_KEY", APIKey: ""},
		"claude-sonnet-4-5": {ID: "claude-sonnet-4-5", APIKeyEnv: "ANTHROPIC_API_KEY", APIKey: ""},
	}}
	engine := newTestEngine(t, []domain.Provider{primary, fallback}, func(cfg *domain.FallbackConfig) {
		cfg.Models = models
	})

	_, err := engine.Run(context.Background(), baseRun(openaiTarget(), anthropicTarget()))
	require.Error(t, err)
	assert.Equal(t, 0, primary.calls)
	assert.Equal(t, 0, fallback.calls, "no target has a usable credential")

	var exhaustedErr *domain.FallbackExhaustedError
	require.ErrorAs(t, err, &exhaustedErr)
	assert.Equal(t, domain.ErrorKindConfig, exhaustedErr.Kind)
	assert.Len(t, exhaustedErr.Attempts, 1, "the walk stopped at the first refusal")
}

// An unsupported capability falls back only if the registry holds a target that
// advertises it.
func TestFallbackEngine_UnsupportedCapability_FallsBackToCapableTarget(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyOpenAI}
	fallback := &fakeVendor{
		family: domain.VendorFamilyAnthropic,
		resp: domain.VendorResponse{
			Completion:   "an image description",
			Usage:        domain.TokenUsage{CostMicros: 30},
			ModelVersion: "claude-sonnet-4-5",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	models := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"gpt-4o":            {ID: "gpt-4o", Capabilities: []string{"chat"}},
		"claude-sonnet-4-5": {ID: "claude-sonnet-4-5", Capabilities: []string{"chat", "image"}},
	}}
	engine := newTestEngine(t, []domain.Provider{primary, fallback}, func(cfg *domain.FallbackConfig) {
		cfg.Models = models
	})

	run := baseRun(openaiTarget(), anthropicTarget())
	run.RequiredCapability = "image"
	out, err := engine.Run(context.Background(), run)
	require.NoError(t, err)
	assert.Equal(t, 0, primary.calls)
	assert.Equal(t, 1, fallback.calls)
	assert.Equal(t, domain.FailureUnsupportedCapability, out.Attempts[0].Class)
}

// No target advertises the capability: the walk stops at the first refusal
// instead of trying a target that cannot serve the request.
func TestFallbackEngine_UnsupportedCapability_NoCapableTargetFailsClosed(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyOpenAI}
	fallback := &fakeVendor{family: domain.VendorFamilyAnthropic}
	models := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"gpt-4o":            {ID: "gpt-4o", Capabilities: []string{"chat"}},
		"claude-sonnet-4-5": {ID: "claude-sonnet-4-5", Capabilities: []string{"chat"}},
	}}
	engine := newTestEngine(t, []domain.Provider{primary, fallback}, func(cfg *domain.FallbackConfig) {
		cfg.Models = models
	})

	run := baseRun(openaiTarget(), anthropicTarget())
	run.RequiredCapability = "image"
	_, err := engine.Run(context.Background(), run)
	require.Error(t, err)

	assert.Equal(t, 0, primary.calls)
	assert.Equal(t, 0, fallback.calls)
	assert.Contains(t, err.Error(), "does not advertise")

	var exhaustedErr *domain.FallbackExhaustedError
	require.ErrorAs(t, err, &exhaustedErr)
	assert.Equal(t, domain.ErrorKindConfig, exhaustedErr.Kind)
}

// A vendor family with no wired adapter is skipped, not fatal: another target
// whose family IS wired can still serve.
func TestFallbackEngine_NoProviderRegistered_SkipsToWiredFamily(t *testing.T) {
	wired := &fakeVendor{
		family: domain.VendorFamilyAnthropic,
		resp: domain.VendorResponse{
			Completion:   "served",
			Usage:        domain.TokenUsage{CostMicros: 40},
			ModelVersion: "claude-sonnet-4-5",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	engine := newTestEngine(t, []domain.Provider{wired})

	out, err := engine.Run(context.Background(), baseRun(openaiTarget(), anthropicTarget()))
	require.NoError(t, err)
	assert.Equal(t, 1, wired.calls)
	assert.Equal(t, "served", out.Response.Completion)
	assert.Equal(t, domain.FailureProvider5xx, out.Attempts[0].Class)
}

// A caller that went away stops the walk immediately: there is nobody left to
// serve, so neither a retry nor a fallback is honest.
func TestFallbackEngine_ContextCancelled_StopsImmediately(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyOpenAI, err: statusError(503)}
	fallback := &fakeVendor{family: domain.VendorFamilyAnthropic, resp: domain.VendorResponse{Completion: "ok"}}
	engine := newTestEngine(t, []domain.Provider{primary, fallback}, func(cfg *domain.FallbackConfig) {
		cfg.MaxSameProviderRetries = 2
	})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := engine.Run(ctx, baseRun(openaiTarget(), anthropicTarget()))
	require.Error(t, err)
	assert.Equal(t, 1, primary.calls)
	assert.Equal(t, 0, fallback.calls)
}

// When every target fails, the error carries the whole trail — the evidence a
// caller needs to attribute the failure without re-deriving the walk.
func TestFallbackEngine_AllTargetsExhausted_CarriesTheTrail(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyOpenAI, err: statusError(503)}
	fallback := &fakeVendor{family: domain.VendorFamilyAnthropic, err: statusError(502)}
	engine := newTestEngine(t, []domain.Provider{primary, fallback})

	_, err := engine.Run(context.Background(), baseRun(openaiTarget(), anthropicTarget()))
	require.Error(t, err)

	var exhaustedErr *domain.FallbackExhaustedError
	require.ErrorAs(t, err, &exhaustedErr)
	assert.Len(t, exhaustedErr.Attempts, 2)
	assert.Equal(t, []string{"openai_byoa:gpt-4o", "anthropic_byoa:claude-sonnet-4-5"}, exhaustedErr.Chain)
	assert.Equal(t, 502, *exhaustedErr.Last.UpstreamStatus,
		"the surfaced status is the LAST target's, mirroring the Python walk")
	assert.Equal(t, domain.ErrorKindInvoke, exhaustedErr.Kind)
	assert.False(t, exhaustedErr.Attempts[0].Billable)
	assert.False(t, exhaustedErr.Attempts[1].Billable)
}

func TestFallbackEngine_RequiresProviders(t *testing.T) {
	_, err := domain.NewFallbackEngine(domain.FallbackConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "at least one Provider")

	_, err = domain.NewFallbackEngine(domain.FallbackConfig{
		Providers:              []domain.Provider{&fakeVendor{family: domain.VendorFamilyOpenAI}},
		MaxSameProviderRetries: -1,
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "MaxSameProviderRetries")

	_, err = domain.NewFallbackEngine(domain.FallbackConfig{
		Providers: []domain.Provider{
			&fakeVendor{family: domain.VendorFamilyOpenAI},
			&fakeVendor{family: domain.VendorFamilyOpenAI},
		},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate Provider")
}

func TestFallbackEngine_RunRequiresTargets(t *testing.T) {
	engine := newTestEngine(t, []domain.Provider{&fakeVendor{family: domain.VendorFamilyOpenAI}})
	_, err := engine.Run(context.Background(), domain.FallbackRun{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no targets")
}

// ----------------------------------------------------------------------------
// Target list construction
// ----------------------------------------------------------------------------

func TestFallbackEngine_Targets_PrimaryFirstThenChainDeduped(t *testing.T) {
	engine := newTestEngine(t, []domain.Provider{&fakeVendor{family: domain.VendorFamilyOpenAI}})

	targets := engine.Targets(
		domain.AgentPolicyFallback{Vendor: domain.VendorFamilyVertexGemini, ResolvedLogicalModelID: "gemini-2.5-pro"},
		[]domain.AgentPolicyFallback{
			{Vendor: domain.VendorFamilyVertexGemini, ResolvedLogicalModelID: "gemini-2.5-pro"}, // duplicate of the primary
			{Vendor: domain.VendorFamilyOpenAI, ResolvedLogicalModelID: "gpt-4o"},
			{Vendor: domain.VendorFamilyOpenAI, ResolvedLogicalModelID: "gpt-4o"}, // duplicate
			{Vendor: domain.VendorFamilyAnthropic, ResolvedLogicalModelID: "claude-sonnet-4-5"},
		},
	)

	require.Len(t, targets, 3)
	assert.Equal(t, "vertex_ai_gemini:gemini-2.5-pro", targets[0].String())
	assert.Equal(t, "openai_byoa:gpt-4o", targets[1].String())
	assert.Equal(t, "anthropic_byoa:claude-sonnet-4-5", targets[2].String())
}

// The engine's retry table is reachable for reporting, and the default table is
// what a caller gets when none is supplied.
func TestFallbackEngine_Matrix_Exposed(t *testing.T) {
	engine := newTestEngine(t, []domain.Provider{&fakeVendor{family: domain.VendorFamilyOpenAI}})
	assert.Equal(t, domain.RetryYes, engine.Matrix().Lookup(domain.FailureRateLimited).Fallback)
}

// ----------------------------------------------------------------------------
// Executor integration: the FallbackEngine is the Executor's only dispatch path
// ----------------------------------------------------------------------------

// Recording ports prove WHERE the database work happens relative to the LLM
// calls. The engine has no database port at all, so the ordering below is
// structural rather than incidental.
type orderRecorder struct{ order []string }

func (r *orderRecorder) add(event string) { r.order = append(r.order, event) }

type recordingClaimer struct {
	rec     *orderRecorder
	claimed bool
}

func (c *recordingClaimer) ClaimDebit(_ context.Context, _, _, _, _ string) (bool, error) {
	c.rec.add("claim")
	return c.claimed, nil
}

type recordingBudget struct{ rec *orderRecorder }

func (b *recordingBudget) GetTenantBudget(_ context.Context, _ string) (*domain.BudgetState, error) {
	b.rec.add("budget_read")
	return &domain.BudgetState{TenantID: "tenant-1", BudgetUSDMicros: 1_000_000, Policy: domain.BudgetPolicyBlock}, nil
}

func (b *recordingBudget) DebitSpent(_ context.Context, _ string, _ int64) error {
	b.rec.add("budget_debit")
	return nil
}

type recordingOutbox struct{ rec *orderRecorder }

func (o *recordingOutbox) EnqueueTokenUsageRecorded(_ context.Context, _ domain.TokenUsageEvent) error {
	o.rec.add("outbox")
	return nil
}

type recordingProvider struct {
	rec    *orderRecorder
	family domain.VendorFamily
	err    error
}

func (p *recordingProvider) Family() domain.VendorFamily { return p.family }

func (p *recordingProvider) Generate(_ context.Context, _ domain.VendorRequest) (domain.VendorResponse, error) {
	p.rec.add("dispatch:" + string(p.family))
	if p.err != nil {
		return domain.VendorResponse{}, p.err
	}
	return domain.VendorResponse{
		Completion:   "ok",
		Usage:        domain.TokenUsage{CostMicros: 100},
		ModelVersion: "m",
		FinishReason: domain.FinishReasonComplete,
	}, nil
}

// capturingManaMeter records the action_code of every debit, so a test can show
// the user's charge does not move with the provider's price.
type capturingManaMeter struct {
	actionCodes []string
}

func (m *capturingManaMeter) Quote(_ context.Context, _, _, _ string) (domain.ManaQuote, error) {
	return domain.ManaQuote{}, nil
}

func (m *capturingManaMeter) Debit(_ context.Context, _, _, actionCode, _ string) (domain.ManaDebit, error) {
	m.actionCodes = append(m.actionCodes, actionCode)
	return domain.ManaDebit{Success: true}, nil
}

// The core accounting invariant: the claim commits, EVERY provider attempt
// happens with no database call in flight, and only then does the settlement
// (debit + outbox) run. If a transaction were held across the LLM call, a
// budget or outbox call would appear between the two dispatches.
func TestExecute_Fallback_NoDBTransactionAcrossTheWalk(t *testing.T) {
	rec := &orderRecorder{}
	primary := &recordingProvider{rec: rec, family: domain.VendorFamilyOpenAI, err: statusError(503)}
	fallback := &recordingProvider{rec: rec, family: domain.VendorFamilyAnthropic}

	executor, _, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Providers = []domain.Provider{primary, fallback}
		cfg.Claims = &recordingClaimer{rec: rec, claimed: true}
		cfg.Budget = &recordingBudget{rec: rec}
		cfg.Outbox = &recordingOutbox{rec: rec}
		cfg.Policies = &fakePolicy{policy: domain.AgentPolicy{
			AgentID:                "qgen_question",
			ResolvedLogicalModelID: domain.LogicalModelID("gpt-4o"),
			Vendor:                 domain.VendorFamilyOpenAI,
			FallbackChain: []domain.AgentPolicyFallback{
				{Vendor: domain.VendorFamilyAnthropic, ResolvedLogicalModelID: domain.LogicalModelID("claude-sonnet-4-5")},
			},
		}}
	})

	req := happyExecuteRequest()
	req.DispatchIdempotencyKey = "wf-turn-1"
	_, err := executor.Execute(context.Background(), req)
	require.NoError(t, err)

	assert.Equal(t, []string{
		"claim",
		"budget_read",
		"dispatch:openai_byoa",
		"dispatch:anthropic_byoa",
		"budget_debit",
		"outbox",
	}, rec.order, "no database call is interleaved with the provider walk")
}

// A POST policy refusal is a decision, not an outage: the platform withholds
// the answer it already has and does NOT ask a second provider for a different
// one.
func TestExecute_PostPolicyBlock_DoesNotFallback(t *testing.T) {
	primary := &fakeVendor{
		family: domain.VendorFamilyVertexGemini,
		resp: domain.VendorResponse{
			Completion:   "a completion the platform refuses",
			Usage:        domain.TokenUsage{CostMicros: 12_500},
			ModelVersion: "gemini-2.5-pro",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	fallback := &fakeVendor{
		family: domain.VendorFamilyOpenAI,
		resp: domain.VendorResponse{
			Completion:   "a second opinion",
			Usage:        domain.TokenUsage{CostMicros: 999},
			ModelVersion: "gpt-4o",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	executor, _, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Providers = []domain.Provider{primary, fallback}
		cfg.Armor = &fakeArmor{
			preVerdict:  domain.ArmorVerdictAllow,
			postVerdict: domain.ArmorVerdictBlock,
		}
		cfg.Policies = &fakePolicy{policy: domain.AgentPolicy{
			AgentID:                "qgen_question",
			ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
			Vendor:                 domain.VendorFamilyVertexGemini,
			ArmorTemplate:          "projects/p/locations/l/templates/t",
			FallbackChain: []domain.AgentPolicyFallback{
				{Vendor: domain.VendorFamilyOpenAI, ResolvedLogicalModelID: domain.LogicalModelID("gpt-4o")},
			},
		}}
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err)

	assert.Equal(t, domain.FinishReasonModelArmorBlock, resp.FinishReason)
	assert.Equal(t, 1, primary.calls)
	assert.Equal(t, 0, fallback.calls,
		"a policy refusal must not be laundered through a second provider")
	assert.Equal(t, []string{"vertex_ai_gemini:gemini-2.5-pro"}, resp.FallbackChain)
}

// The ledger records the winner's cost and nothing else — not the failed
// attempt's partial output.
func TestExecute_Fallback_OnlyTheWinningCostIsDebited(t *testing.T) {
	primary := &scriptedProvider{
		family: domain.VendorFamilyOpenAI,
		results: []scriptedResult{{
			resp: domain.VendorResponse{
				Usage: domain.TokenUsage{OutputTokens: 2_000, CostMicros: 5_000},
			},
			err: statusError(429),
		}},
	}
	fallback := &fakeVendor{
		family: domain.VendorFamilyAnthropic,
		resp: domain.VendorResponse{
			Completion:   "the answer",
			Usage:        domain.TokenUsage{InputTokens: 10, OutputTokens: 2_500, CostMicros: 3_000},
			ModelVersion: "claude-sonnet-4-5",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	executor, _, _, budget, outbox, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Providers = []domain.Provider{primary, fallback}
		cfg.Policies = &fakePolicy{policy: domain.AgentPolicy{
			AgentID:                "qgen_question",
			ResolvedLogicalModelID: domain.LogicalModelID("gpt-4o"),
			Vendor:                 domain.VendorFamilyOpenAI,
			FallbackChain: []domain.AgentPolicyFallback{
				{Vendor: domain.VendorFamilyAnthropic, ResolvedLogicalModelID: domain.LogicalModelID("claude-sonnet-4-5")},
			},
		}}
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err)

	assert.Equal(t, int64(3_000), resp.Usage.CostMicros)
	assert.Equal(t, int64(3_000), budget.debits,
		"the tenant is debited the winner's cost, not the failed attempt's partial cost")
	require.Len(t, outbox.events, 1)
	assert.Equal(t, int64(3_000), outbox.events[0].CostMicros)
	assert.Equal(t, "claude-sonnet-4-5", outbox.events[0].ModelID)
	assert.Equal(t, []string{"openai_byoa:gpt-4o", "anthropic_byoa:claude-sonnet-4-5"}, resp.FallbackChain)
}

// Provider cost and the user's mana charge are different values: the provider
// price follows whichever vendor answered, while the mana action_code — the
// user-facing price — does not move at all.
func TestExecute_ProviderCostAndManaChargeAreSeparate(t *testing.T) {
	run := func(t *testing.T, providerCost int64) (int64, string) {
		t.Helper()
		primary := &fakeVendor{family: domain.VendorFamilyOpenAI, err: statusError(503)}
		fallback := &fakeVendor{
			family: domain.VendorFamilyAnthropic,
			resp: domain.VendorResponse{
				Completion:   "ok",
				Usage:        domain.TokenUsage{CostMicros: providerCost},
				ModelVersion: "claude-sonnet-4-5",
				FinishReason: domain.FinishReasonComplete,
			},
		}
		mana := &capturingManaMeter{}
		executor, _, _, _, outbox, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
			cfg.Providers = []domain.Provider{primary, fallback}
			cfg.Mana = mana
			cfg.Policies = &fakePolicy{policy: domain.AgentPolicy{
				AgentID:                "qgen_question",
				ResolvedLogicalModelID: domain.LogicalModelID("gpt-4o"),
				Vendor:                 domain.VendorFamilyOpenAI,
				FallbackChain: []domain.AgentPolicyFallback{
					{Vendor: domain.VendorFamilyAnthropic, ResolvedLogicalModelID: domain.LogicalModelID("claude-sonnet-4-5")},
				},
			}}
		})

		req := happyExecuteRequest()
		req.ActionCode = "qgen_question_turn"
		_, err := executor.Execute(context.Background(), req)
		require.NoError(t, err)

		require.Len(t, outbox.events, 1)
		require.Len(t, mana.actionCodes, 1)
		return outbox.events[0].CostMicros, mana.actionCodes[0]
	}

	cheapCost, cheapAction := run(t, 100)
	dearCost, dearAction := run(t, 900_000)

	assert.Equal(t, int64(100), cheapCost)
	assert.Equal(t, int64(900_000), dearCost)
	assert.Equal(t, "qgen_question_turn", cheapAction)
	assert.Equal(t, cheapAction, dearAction,
		"the user's mana charge does not move when the winning provider's price does")
}

// ExecutorConfig.FallbackRetries reaches the engine: a 503 is retried on the
// same provider before the walk would advance.
func TestExecute_SameProviderRetry_WiredThroughConfig(t *testing.T) {
	primary := &scriptedProvider{
		family: domain.VendorFamilyVertexGemini,
		results: []scriptedResult{
			{err: statusError(503)},
			{resp: domain.VendorResponse{
				Completion:   "recovered",
				Usage:        domain.TokenUsage{CostMicros: 50},
				ModelVersion: "gemini-2.5-pro",
				FinishReason: domain.FinishReasonComplete,
			}},
		},
	}
	executor, _, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Providers = []domain.Provider{primary}
		cfg.FallbackRetries = 1
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err)
	assert.Equal(t, "recovered", resp.Completion)
	assert.Equal(t, 2, primary.calls)
}

// The Executor consumes the engine's outcome, so the response carries the
// WINNING target rather than the intended one.
func TestExecute_Fallback_ResponseCarriesTheWinningTarget(t *testing.T) {
	primary := &fakeVendor{family: domain.VendorFamilyVertexGemini, err: statusError(503)}
	fallback := &fakeVendor{
		family: domain.VendorFamilyOpenAI,
		resp: domain.VendorResponse{
			Completion:   "from the fallback",
			Usage:        domain.TokenUsage{CostMicros: 1_000},
			ModelVersion: "gpt-4o-mini",
			FinishReason: domain.FinishReasonComplete,
		},
	}
	executor, _, _, _, _, _ := newTestExecutor(t, func(cfg *domain.ExecutorConfig) {
		cfg.Providers = []domain.Provider{primary, fallback}
		cfg.Policies = &fakePolicy{policy: domain.AgentPolicy{
			AgentID:                "qgen_question",
			ResolvedLogicalModelID: domain.LogicalModelID("gemini-2.5-pro"),
			Vendor:                 domain.VendorFamilyVertexGemini,
			FallbackChain: []domain.AgentPolicyFallback{
				{Vendor: domain.VendorFamilyOpenAI, ResolvedLogicalModelID: domain.LogicalModelID("gpt-4o-mini")},
			},
		}}
	})

	resp, err := executor.Execute(context.Background(), happyExecuteRequest())
	require.NoError(t, err)
	assert.Equal(t, "from the fallback", resp.Completion)
	assert.Equal(t, string(domain.VendorFamilyOpenAI), resp.Vendor)
	assert.Equal(t, "gpt-4o-mini", resp.ModelVersion)
	assert.Equal(t, domain.AccountingAccounted, resp.AccountingState)
}
