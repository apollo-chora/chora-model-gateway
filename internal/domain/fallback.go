// fallback.go — the cross-provider fallback engine and the explicit retry
// matrix that governs it.
//
// The engine exists to make one question answerable without reading code:
// "this attempt failed — do we retry the same provider, move to the next
// target, or stop?" That answer is a table (RetryMatrix), not a chain of
// if-statements scattered through the dispatch loop, because the decision is
// an accounting and safety policy rather than an implementation detail. The
// Python reference (app/service.py) walked its chain on ANY exception; this
// port keeps the walk but classifies the failure first, so a request the
// provider will reject identically every time (a malformed payload) does not
// burn the whole chain, and a policy refusal never looks like an outage.
//
// Two rules the engine enforces structurally:
//
//  1. ACCOUNTING. Only the WINNING attempt is billable. A provider that
//     produced 2000 tokens and then errored has NOT served the turn: its
//     partial output is discarded, its cost is not charged, and the ledger
//     records the one provider that actually answered. FallbackOutcome carries
//     the winning attempt's usage and nothing else; the failed attempts' usage
//     survives only on the audit trail (FallbackAttempt.Usage) for diagnosis.
//
//  2. NO DATABASE TRANSACTION ACROSS AN LLM CALL. The engine holds no
//     database port at all — not a claimer, not a budget repo, not an outbox.
//     It cannot open a transaction because it has nothing to open one with.
//     The Executor takes the idempotency claim before Run and settles after
//     it, each in its own short transaction.
//
// Provider cost and the user's mana charge are deliberately different values.
// See ProviderCharge.
package domain

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"
)

// =============================================================================
// Failure classification
// =============================================================================

// FailureClass names WHY an attempt failed. The class — not the call site — is
// what the retry matrix reasons about, so a timeout raised by an HTTP client
// and a timeout raised by a test fake classify identically.
type FailureClass int

const (
	// FailureUnknown is the zero value and the fail-closed bucket: an
	// unclassified failure is never retried and never falls back. Anything the
	// gateway cannot explain must not be worked around silently.
	FailureUnknown FailureClass = iota

	// FailureTimeout — the provider produced no response within the deadline
	// (context deadline exceeded, or a transport timeout).
	FailureTimeout

	// FailureRateLimited — the provider answered 429.
	FailureRateLimited

	// FailureProvider5xx — a provider-side infrastructure failure: an explicit
	// 5xx, a connection refused, a DNS failure, a truncated response, or a
	// target whose vendor adapter is not registered. These are indistinguishable
	// to the gateway and all warrant the same action.
	FailureProvider5xx

	// FailureInvalidCredentials — the provider rejected the credential (401/403)
	// or the registry's credential reference resolves to an empty value.
	FailureInvalidCredentials

	// FailureMalformedRequest — the provider rejected the request payload
	// (400/422). Every provider will reject the same payload the same way, so
	// walking the chain cannot help.
	FailureMalformedRequest

	// FailureUnsupportedCapability — the target does not offer what the request
	// needs: the registry says it lacks the capability, or the provider answered
	// 404 (the model is not there).
	FailureUnsupportedCapability

	// FailurePrePolicyRejection — a PRE-LLM policy hop (Cloud Model Armor PRE)
	// refused the input. Governance, not infrastructure.
	FailurePrePolicyRejection

	// FailurePostPolicyRejection — a POST-LLM policy hop (Cloud Model Armor
	// POST) refused the model's output. Governance, not infrastructure: the
	// model answered and the platform chose to withhold the answer. Asking a
	// different provider to re-answer is a policy decision, not a recovery.
	FailurePostPolicyRejection

	// FailureBudgetExhausted — the tenant's LLM budget refused the call.
	FailureBudgetExhausted

	// FailureLedgerFailure — the usage ledger could not be written.
	FailureLedgerFailure

	// FailureInternalInvariant — the gateway violated one of its own
	// invariants. Never retried and never worked around: a second provider
	// would hit the same broken assumption.
	FailureInternalInvariant
)

// String returns the machine token used in logs, span attributes and tests.
func (c FailureClass) String() string {
	switch c {
	case FailureTimeout:
		return "timeout"
	case FailureRateLimited:
		return "rate_limited"
	case FailureProvider5xx:
		return "provider_5xx"
	case FailureInvalidCredentials:
		return "invalid_credentials"
	case FailureMalformedRequest:
		return "malformed_request"
	case FailureUnsupportedCapability:
		return "unsupported_capability"
	case FailurePrePolicyRejection:
		return "pre_policy_rejection"
	case FailurePostPolicyRejection:
		return "post_policy_rejection"
	case FailureBudgetExhausted:
		return "budget_exhausted"
	case FailureLedgerFailure:
		return "ledger_failure"
	case FailureInternalInvariant:
		return "internal_invariant_failure"
	default:
		return "unknown"
	}
}

// =============================================================================
// The retry matrix
// =============================================================================

// RetryDecision is the three-valued answer the matrix gives to each question.
//
// The third value is load-bearing. "yes" means the failure is an
// infrastructure fault that always warrants the action. "maybe" means the
// action is permitted but the engine must confirm the premise first — for a
// fallback, that the registry actually has a target which can serve the
// request, so the chain is not walked into an identical failure. Collapsing
// "maybe" into "yes" would turn a registry lookup into a blind retry.
type RetryDecision int

const (
	// RetryNo — the action is forbidden for this failure class.
	RetryNo RetryDecision = iota
	// RetryMaybe — the action is permitted once its premise is confirmed.
	RetryMaybe
	// RetryYes — the action is unconditionally warranted.
	RetryYes
)

func (d RetryDecision) String() string {
	switch d {
	case RetryMaybe:
		return "maybe"
	case RetryYes:
		return "yes"
	default:
		return "no"
	}
}

// RetryRule is one row of the retry matrix: what may happen after an attempt
// fails with Class.
type RetryRule struct {
	// Class is the failure this rule covers.
	Class FailureClass
	// SameProvider is whether the SAME provider may be attempted again. The
	// engine additionally requires a non-zero same-provider retry budget
	// (FallbackConfig.MaxSameProviderRetries), which is 0 by default: retrying
	// a rate-limited provider without an operator opting in amplifies the load
	// that caused the 429.
	SameProvider RetryDecision
	// Fallback is whether the engine may advance to the next target.
	Fallback RetryDecision
	// Rationale records why the row reads the way it does, so the policy can
	// be reviewed without re-deriving it from first principles.
	Rationale string
}

// RetryMatrix is the explicit table of retry decisions. The zero value is
// empty and fails closed (no retry, no fallback); use DefaultRetryMatrix for
// the ratified policy.
type RetryMatrix struct {
	rules map[FailureClass]RetryRule
}

// DefaultRetryMatrix returns the ratified policy table.
//
// The asymmetry is the whole point. Infrastructure failures (timeout, 429,
// 5xx) fall back unconditionally. Failures the platform decided on — a policy
// rejection, a budget refusal, a ledger failure, an internal invariant — never
// fall back, because a second provider cannot change a decision the platform
// made, and trying would spend money to produce the same answer. A malformed
// payload never falls back either: it is the caller's request, and every
// provider will reject it identically.
func DefaultRetryMatrix() RetryMatrix {
	rules := []RetryRule{
		{
			Class:        FailureTimeout,
			SameProvider: RetryMaybe,
			Fallback:     RetryYes,
			Rationale:    "no response within the deadline is an infrastructure fault; a retry may clear it and another provider is an independent chance",
		},
		{
			Class:        FailureRateLimited,
			SameProvider: RetryMaybe,
			Fallback:     RetryYes,
			Rationale:    "429 is a capacity signal, not a verdict; falling back spreads the load, and an immediate same-provider retry is opt-in only",
		},
		{
			Class:        FailureProvider5xx,
			SameProvider: RetryMaybe,
			Fallback:     RetryYes,
			Rationale:    "the provider is unhealthy; another target is an independent chance to serve the turn",
		},
		{
			Class:        FailureInvalidCredentials,
			SameProvider: RetryNo,
			Fallback:     RetryMaybe,
			Rationale:    "the same key cannot start working, so a same-provider retry is pure waste; another target is worth trying only if the registry shows it has a credential that resolves",
		},
		{
			Class:        FailureMalformedRequest,
			SameProvider: RetryNo,
			Fallback:     RetryNo,
			Rationale:    "the payload is the caller's and every provider rejects it identically; walking the chain burns money to produce the same refusal",
		},
		{
			Class:        FailureUnsupportedCapability,
			SameProvider: RetryNo,
			Fallback:     RetryMaybe,
			Rationale:    "the target cannot serve the request, so retrying it is pointless; fall back only if the registry holds a target that advertises the capability",
		},
		{
			Class:        FailurePrePolicyRejection,
			SameProvider: RetryNo,
			Fallback:     RetryNo,
			Rationale:    "a governance refusal is a decision, not an outage; re-asking cannot un-refuse the input and must not be laundered through another provider",
		},
		{
			Class:        FailurePostPolicyRejection,
			SameProvider: RetryNo,
			Fallback:     RetryNo,
			Rationale:    "the model answered and the platform withheld the answer; trying another provider to obtain a different answer is a policy decision, never an automatic recovery",
		},
		{
			Class:        FailureBudgetExhausted,
			SameProvider: RetryNo,
			Fallback:     RetryNo,
			Rationale:    "the tenant is out of budget; serving the same turn from another provider spends the money the refusal was protecting",
		},
		{
			Class:        FailureLedgerFailure,
			SameProvider: RetryNo,
			Fallback:     RetryNo,
			Rationale:    "usage could not be recorded; retrying would risk an unrecorded second spend on top of an unrecorded first",
		},
		{
			Class:        FailureInternalInvariant,
			SameProvider: RetryNo,
			Fallback:     RetryNo,
			Rationale:    "the gateway broke its own contract; a second provider runs into the same broken assumption",
		},
		{
			Class:        FailureUnknown,
			SameProvider: RetryNo,
			Fallback:     RetryNo,
			Rationale:    "an unclassified failure is not understood well enough to be worked around; fail closed",
		},
	}

	m := RetryMatrix{rules: make(map[FailureClass]RetryRule, len(rules))}
	for _, r := range rules {
		m.rules[r.Class] = r
	}
	return m
}

// Lookup returns the rule for a class. An empty matrix, or a class with no
// row, fails closed: no same-provider retry, no fallback.
func (m RetryMatrix) Lookup(class FailureClass) RetryRule {
	if m.rules != nil {
		if r, ok := m.rules[class]; ok {
			return r
		}
	}
	return RetryRule{
		Class:        class,
		SameProvider: RetryNo,
		Fallback:     RetryNo,
		Rationale:    "no rule for this class; fail closed",
	}
}

// Rules returns the matrix rows ordered by class, for audit output and tests.
func (m RetryMatrix) Rules() []RetryRule {
	out := make([]RetryRule, 0, len(m.rules))
	for c := FailureUnknown; c <= FailureInternalInvariant; c++ {
		if r, ok := m.rules[c]; ok {
			out = append(out, r)
		}
	}
	return out
}

// =============================================================================
// Classification
// =============================================================================

// ClassifyError maps a dispatch or preflight error to a FailureClass.
//
// The order of the checks matters: a typed gateway error is classified by its
// type before the generic HTTP-status path, so a capability refusal that
// happens to carry a 404 does not read as a malformed request.
func ClassifyError(err error) FailureClass {
	if err == nil {
		return FailureUnknown
	}

	var capErr *CapabilityError
	if errors.As(err, &capErr) {
		return FailureUnsupportedCapability
	}
	var credErr *CredentialError
	if errors.As(err, &credErr) {
		return FailureInvalidCredentials
	}
	var noProvider *NoProviderError
	if errors.As(err, &noProvider) {
		// A family with no wired adapter is an infrastructure gap, not a
		// caller error: another target whose family IS wired can still serve.
		return FailureProvider5xx
	}
	var cfgErr *ConfigError
	if errors.As(err, &cfgErr) {
		// A registry that cannot resolve a named target is a configuration
		// fault. Falling back would hide the misconfiguration, so fail closed.
		return FailureUnknown
	}

	if errors.Is(err, context.DeadlineExceeded) {
		return FailureTimeout
	}
	if errors.Is(err, context.Canceled) {
		// The caller went away. There is nobody to serve, so this is neither
		// retried nor fallen back; the engine also stops on ctx.Err().
		return FailureUnknown
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return FailureTimeout
	}
	if status := UpstreamStatusOf(err); status != nil {
		return classifyStatus(*status)
	}
	// A transport failure with no status and no timeout (connection refused,
	// DNS failure, truncated body) is a provider-side infrastructure fault.
	return FailureProvider5xx
}

// classifyStatus maps a provider HTTP status to a failure class.
func classifyStatus(status int) FailureClass {
	switch {
	case status == 408:
		return FailureTimeout
	case status == 429:
		return FailureRateLimited
	case status == 401 || status == 403:
		return FailureInvalidCredentials
	case status == 404:
		// The model is not at this provider. The registry and the provider
		// disagree, which is exactly what another target can fix.
		return FailureUnsupportedCapability
	case status >= 500:
		return FailureProvider5xx
	case status >= 400:
		return FailureMalformedRequest
	default:
		return FailureUnknown
	}
}

// ClassifyFinishReason maps a governance short-circuit's finish reason to a
// failure class, so a caller that never reaches the engine (an Armor or budget
// refusal) still consults the same table when asking "should this have been
// retried?". The Executor uses it to keep the matrix the single source of
// truth for that question.
func ClassifyFinishReason(reason FinishReason) FailureClass {
	switch reason {
	case FinishReasonModelArmorBlock:
		// The gateway cannot tell PRE from POST from the finish reason alone;
		// both rows read the same way (never fall back), so the ambiguity is
		// harmless here.
		return FailurePostPolicyRejection
	case FinishReasonBudgetBlock:
		return FailureBudgetExhausted
	case FinishReasonVendorError:
		return FailureProvider5xx
	default:
		return FailureUnknown
	}
}

// =============================================================================
// Preflight refusals
// =============================================================================

// CapabilityError is a per-target preflight refusal: the target does not
// advertise the capability the request needs. It is a gateway configuration
// problem — the chain names a model that cannot serve the request — so the
// Executor maps it to ErrorKindConfig.
type CapabilityError struct {
	Model      LogicalModelID
	Capability string
}

func (e *CapabilityError) Error() string {
	return fmt.Sprintf("model %q does not advertise the %q capability", e.Model, e.Capability)
}

// CredentialError is a per-target preflight refusal: either the registry's
// credential reference is set but resolves to an empty value, or the entry
// declares no reference at all while pointing at a provider that authenticates
// every request — so the call would go out unauthenticated. ErrorKindConfig,
// like CapabilityError. The credential VALUE is never carried (or logged).
type CredentialError struct {
	Model     LogicalModelID
	APIKeyEnv string
	// Detail replaces the default message for the complementary refusal: an
	// entry with no credential reference dispatched to an authenticated host.
	Detail string
}

func (e *CredentialError) Error() string {
	if e.Detail != "" {
		return e.Detail
	}
	return fmt.Sprintf("credential reference %q is set but resolves to an empty value", e.APIKeyEnv)
}

// NoProviderError is a per-target preflight refusal: no vendor adapter is
// registered for the target's family. Unlike the two above it is an
// infrastructure gap rather than a caller mistake, so the matrix lets the walk
// continue to a target whose family IS wired.
type NoProviderError struct {
	Family VendorFamily
}

func (e *NoProviderError) Error() string {
	return fmt.Sprintf("no Provider registered for family %q", e.Family)
}

// =============================================================================
// Targets, attempts and outcome
// =============================================================================

// FallbackTarget is one candidate in the walk: the (vendor, logical model)
// pair the engine dispatches to.
type FallbackTarget struct {
	Vendor         VendorFamily
	LogicalModelID LogicalModelID
}

// String renders the target as "vendor:model" — the exact form the response's
// fallback chain and the usage ledger record.
func (t FallbackTarget) String() string {
	return fmt.Sprintf("%s:%s", t.Vendor, t.LogicalModelID)
}

// FallbackAttempt records one dispatch attempt. The trail is the evidence a
// consumer needs to attribute cost and latency to the real (rather than the
// intended) dispatch.
type FallbackAttempt struct {
	// Target is the candidate this attempt dispatched to.
	Target FallbackTarget

	// Class is why the attempt failed. Meaningful only when Billable is false;
	// it is FailureUnknown on the winning attempt.
	Class FailureClass

	// Err is the attempt's own error, nil on success.
	Err error

	// UpstreamStatus is the provider's HTTP status when the failure carried
	// one. Nil for a transport or preflight failure.
	UpstreamStatus *int

	// SameProviderRetries is how many extra same-provider attempts this target
	// consumed after its first failure.
	SameProviderRetries int

	// Usage is the attempt's OWN usage. On the winning attempt it is the
	// billable usage. On a failed attempt it is whatever partial usage the
	// provider reported before it errored — recorded for diagnosis and NEVER
	// charged.
	Usage TokenUsage

	// Billable is true for exactly one attempt: the one that returned a
	// completion the gateway will pay for. Every failed attempt is false,
	// including one that reported partial token usage before it errored.
	Billable bool
}

// ProviderCharge is what the gateway owes the WINNING provider for one call,
// derived from that provider's own reported token usage and price table.
//
// It is deliberately NOT the user's mana charge. The mana charge is priced by
// the identity catalogue per action_code and does not move when the walk lands
// on a cheaper or dearer provider: a turn served by a fallback provider still
// costs the learner the same mana, while the ledger still records the
// provider's real cost. Keeping them separate is what stops a provider-price
// change from silently re-pricing a user-facing action.
type ProviderCharge struct {
	// CostMicros is the provider's cost in USD micros.
	CostMicros int64
	// Vendor and ModelVersion identify the provider that was actually paid —
	// the winning target, not the intended one.
	Vendor       string
	ModelVersion string
}

// FallbackOutcome is the result of one governed walk that succeeded.
type FallbackOutcome struct {
	// Response is the winning provider's response.
	Response VendorResponse

	// Target is the candidate that served the turn.
	Target FallbackTarget

	// Chain is one "vendor:model" entry per target attempted, in order.
	Chain []string

	// Attempts is the full per-attempt audit trail, in order.
	Attempts []FallbackAttempt

	// ProviderCharge is what the gateway owes the winning provider. A failed
	// attempt's partial output is not billable and contributes nothing here.
	ProviderCharge ProviderCharge
}

// FallbackExhaustedError is returned when every target in the walk failed. It
// carries the full attempt trail, so the caller builds its terminal error from
// evidence rather than re-deriving it.
type FallbackExhaustedError struct {
	// Attempts is the full audit trail, in order.
	Attempts []FallbackAttempt

	// Chain is one "vendor:model" entry per attempted target.
	Chain []string

	// Last is the final attempt — the one whose failure is surfaced.
	Last FallbackAttempt

	// Kind is the gateway error taxonomy the transport maps to a status code.
	Kind ErrorKind
}

func (e *FallbackExhaustedError) Error() string {
	if e.Last.Err != nil {
		return fmt.Sprintf("all %d provider attempts failed; last target %s: %v", len(e.Attempts), e.Last.Target, e.Last.Err)
	}
	return fmt.Sprintf("all %d provider attempts failed; last target %s", len(e.Attempts), e.Last.Target)
}

// Unwrap exposes the last attempt's error so errors.Is/As see through the
// walk's summary.
func (e *FallbackExhaustedError) Unwrap() error { return e.Last.Err }

// =============================================================================
// The engine
// =============================================================================

// FallbackConfig groups the engine's dependencies. Providers is REQUIRED; a
// model resolver is optional (without one the capability / credential /
// output-ceiling preflight is skipped and the target list is the only permit).
type FallbackConfig struct {
	// Providers is one Provider per VendorFamily. REQUIRED — at least one.
	Providers []Provider

	// Models is the model-registry resolver. OPTIONAL; nil skips the per-target
	// preflight.
	Models ModelResolver

	// Matrix overrides the ratified retry table. Zero value uses
	// DefaultRetryMatrix.
	Matrix RetryMatrix

	// MaxSameProviderRetries is the number of ADDITIONAL attempts on the same
	// provider after a retryable failure. 0 — the default — disables
	// same-provider retry entirely, so a "maybe" row reads as "no" until an
	// operator opts in. Negative values are rejected.
	MaxSameProviderRetries int

	// RetryBackoff is the delay before each same-provider retry. 0 retries
	// immediately.
	RetryBackoff time.Duration

	// Sleep is the backoff sleeper. OPTIONAL; nil uses a context-aware timer.
	// Tests substitute a no-op so a backoff does not slow the suite down.
	Sleep func(ctx context.Context, d time.Duration) error
}

// FallbackRun is one governed dispatch handed to the engine.
type FallbackRun struct {
	// Targets is the ordered candidate list; Targets[0] is the primary.
	// REQUIRED — an empty list is a programming error.
	Targets []FallbackTarget

	// Base is the canonical vendor request shared by every target. The engine
	// overrides Base.Vendor and Base.LogicalModelID per target, and sets
	// Base.GenerationConfig to the per-target clamped value.
	Base VendorRequest

	// GenerationConfig is the caller's generation parameters BEFORE per-target
	// clamping. A fallback to a model with a lower output ceiling re-clamps.
	GenerationConfig map[string]any

	// RequiredCapability is the capability every target must advertise
	// ("chat" / "image" / "web_search"). Empty skips the check. Ignored when
	// no model resolver is wired.
	RequiredCapability string
}

// FallbackEngine orchestrates the walk across providers: it builds the target
// list, runs the per-target preflight, dispatches with the retry matrix's
// same-provider policy, and decides whether a failure advances to the next
// target or stops.
//
// The engine holds NO state across runs and NO database port. It is safe to
// call from multiple goroutines.
type FallbackEngine struct {
	providers map[VendorFamily]Provider
	models    ModelResolver
	matrix    RetryMatrix

	maxSameProviderRetries int
	retryBackoff           time.Duration
	sleep                  func(ctx context.Context, d time.Duration) error
}

// NewFallbackEngine constructs an engine. Returns an error if no provider is
// supplied — fail-loud, never a silent single-provider engine.
func NewFallbackEngine(cfg FallbackConfig) (*FallbackEngine, error) {
	if len(cfg.Providers) == 0 {
		return nil, errors.New("model gateway: FallbackEngine requires at least one Provider")
	}
	if cfg.MaxSameProviderRetries < 0 {
		return nil, errors.New("model gateway: FallbackEngine MaxSameProviderRetries must be >= 0")
	}

	providers := make(map[VendorFamily]Provider, len(cfg.Providers))
	for _, p := range cfg.Providers {
		if p == nil {
			return nil, errors.New("model gateway: nil Provider in FallbackConfig.Providers")
		}
		fam := p.Family()
		if fam == "" {
			return nil, errors.New("model gateway: Provider.Family() returned empty")
		}
		if _, dup := providers[fam]; dup {
			return nil, fmt.Errorf("model gateway: duplicate Provider for family %q", fam)
		}
		providers[fam] = p
	}

	matrix := cfg.Matrix
	if matrix.rules == nil {
		matrix = DefaultRetryMatrix()
	}
	sleep := cfg.Sleep
	if sleep == nil {
		sleep = sleepWithContext
	}

	return &FallbackEngine{
		providers:              providers,
		models:                 cfg.Models,
		matrix:                 matrix,
		maxSameProviderRetries: cfg.MaxSameProviderRetries,
		retryBackoff:           cfg.RetryBackoff,
		sleep:                  sleep,
	}, nil
}

// Matrix exposes the engine's retry table so a caller can report or assert the
// policy it is running under.
func (e *FallbackEngine) Matrix() RetryMatrix { return e.matrix }

// Targets builds the ordered, de-duplicated candidate list: the primary first,
// then the policy's fallback chain. A duplicate (vendor, model) pair is
// dropped — retrying the identical pair adds nothing but latency and load.
func (e *FallbackEngine) Targets(primary AgentPolicyFallback, chain []AgentPolicyFallback) []FallbackTarget {
	targets := make([]FallbackTarget, 0, 1+len(chain))
	seen := make(map[FallbackTarget]struct{}, 1+len(chain))
	add := func(vendor VendorFamily, model LogicalModelID) {
		t := FallbackTarget{Vendor: vendor, LogicalModelID: model}
		if _, dup := seen[t]; dup {
			return
		}
		seen[t] = struct{}{}
		targets = append(targets, t)
	}
	add(primary.Vendor, primary.ResolvedLogicalModelID)
	for _, fb := range chain {
		add(fb.Vendor, fb.ResolvedLogicalModelID)
	}
	return targets
}

// Run walks the targets and returns the first successful outcome.
//
// On success the returned outcome carries ONLY the winning attempt's usage:
// the turn is charged once, to the provider that actually served it. When
// every target fails, the returned *FallbackExhaustedError carries the whole
// trail.
//
// No database transaction is held — or possible — across the walk.
func (e *FallbackEngine) Run(ctx context.Context, run FallbackRun) (FallbackOutcome, error) {
	if len(run.Targets) == 0 {
		return FallbackOutcome{}, errors.New("model gateway: fallback run has no targets")
	}

	outcome := FallbackOutcome{Chain: make([]string, 0, len(run.Targets))}

	for idx, target := range run.Targets {
		outcome.Chain = append(outcome.Chain, target.String())

		req, prepErr := e.prepareTarget(ctx, run, target)
		if prepErr != nil {
			attempt := FallbackAttempt{
				Target:         target,
				Class:          ClassifyError(prepErr),
				Err:            prepErr,
				UpstreamStatus: UpstreamStatusOf(prepErr),
			}
			outcome.Attempts = append(outcome.Attempts, attempt)
			if !e.fallbackPermitted(ctx, run, idx, attempt.Class) {
				return FallbackOutcome{}, exhausted(outcome, attempt)
			}
			continue
		}

		resp, lastErr, retries := e.dispatch(ctx, target, req)

		if lastErr == nil {
			attempt := FallbackAttempt{
				Target:              target,
				SameProviderRetries: retries,
				Usage:               resp.Usage,
				Billable:            true,
			}
			outcome.Attempts = append(outcome.Attempts, attempt)
			outcome.Response = resp
			outcome.Target = target
			outcome.ProviderCharge = ProviderCharge{
				CostMicros:   resp.Usage.CostMicros,
				Vendor:       string(target.Vendor),
				ModelVersion: resp.ModelVersion,
			}
			return outcome, nil
		}

		class := ClassifyError(lastErr)
		attempt := FallbackAttempt{
			Target:              target,
			Class:               class,
			Err:                 lastErr,
			UpstreamStatus:      UpstreamStatusOf(lastErr),
			SameProviderRetries: retries,
			// resp may carry the partial usage the provider reported before it
			// errored. It is recorded as evidence and contributes nothing to
			// ProviderCharge: the turn was not served.
			Usage: resp.Usage,
		}
		outcome.Attempts = append(outcome.Attempts, attempt)
		if !e.fallbackPermitted(ctx, run, idx, class) {
			return FallbackOutcome{}, exhausted(outcome, attempt)
		}
	}

	// Unreachable in practice — fallbackPermitted is always false for the last
	// target, so every iteration returns from inside the loop. The compiler
	// needs a terminating statement after the loop, and keeping the trail-based
	// error here means a future change to that invariant degrades to a correct
	// summary rather than a nil return.
	last := outcome.Attempts[len(outcome.Attempts)-1]
	return FallbackOutcome{}, exhausted(outcome, last)
}

// dispatch runs one target, applying the matrix's same-provider retry policy.
// It returns the last response (which may carry partial usage), the last
// error, and how many extra same-provider attempts were consumed.
func (e *FallbackEngine) dispatch(ctx context.Context, target FallbackTarget, req VendorRequest) (VendorResponse, error, int) {
	var resp VendorResponse
	var lastErr error
	retries := 0

	for {
		resp, lastErr = e.generate(ctx, target, req)
		if lastErr == nil {
			return resp, nil, retries
		}

		// A cancelled or expired caller stops the walk immediately: there is
		// nobody left to serve, so neither a retry nor a fallback is honest.
		if ctx.Err() != nil {
			return resp, lastErr, retries
		}

		rule := e.matrix.Lookup(ClassifyError(lastErr))
		if rule.SameProvider == RetryNo || retries >= e.maxSameProviderRetries {
			return resp, lastErr, retries
		}
		retries++
		if e.retryBackoff > 0 {
			if serr := e.sleep(ctx, e.retryBackoff); serr != nil {
				return resp, lastErr, retries
			}
		}
	}
}

// generate looks up the target's adapter and calls it. A family with no wired
// adapter is an infrastructure gap, not a panic.
func (e *FallbackEngine) generate(ctx context.Context, target FallbackTarget, req VendorRequest) (VendorResponse, error) {
	client, ok := e.providers[target.Vendor]
	if !ok {
		return VendorResponse{}, &NoProviderError{Family: target.Vendor}
	}
	return client.Generate(ctx, req)
}

// prepareTarget runs the per-target preflight and materialises the vendor
// request: registry metadata (capability, credential), the per-target output
// ceiling clamp, and the adapter lookup. Every refusal is a typed error so the
// matrix can classify it without string matching.
func (e *FallbackEngine) prepareTarget(ctx context.Context, run FallbackRun, target FallbackTarget) (VendorRequest, error) {
	req := run.Base
	req.Vendor = target.Vendor
	req.LogicalModelID = target.LogicalModelID
	req.GenerationConfig = run.GenerationConfig

	if _, ok := e.providers[target.Vendor]; !ok {
		return VendorRequest{}, &NoProviderError{Family: target.Vendor}
	}

	if e.models != nil {
		info, err := e.models.Resolve(ctx, target.LogicalModelID)
		if err != nil {
			return VendorRequest{}, &ConfigError{
				Detail: fmt.Sprintf("model %q is not in the registry", target.LogicalModelID),
				Inner:  err,
			}
		}
		if run.RequiredCapability != "" && !info.Supports(run.RequiredCapability) {
			return VendorRequest{}, &CapabilityError{
				Model:      target.LogicalModelID,
				Capability: run.RequiredCapability,
			}
		}
		if info.APIKeyEnv != "" && info.APIKey == "" {
			return VendorRequest{}, &CredentialError{
				Model:     target.LogicalModelID,
				APIKeyEnv: info.APIKeyEnv,
			}
		}
		// The output ceiling is per-target: a fallback to a model with a lower
		// ceiling must re-clamp.
		if info.MaxOutputTokens > 0 {
			if cfg := clampMaxTokens(run.GenerationConfig, info.MaxOutputTokens); cfg != nil {
				req.GenerationConfig = cfg
			}
		}
	}

	return req, nil
}

// fallbackPermitted answers "may the walk advance past target idx after a
// class failure?" It is the matrix lookup plus the two premises a "maybe" row
// requires: a next target must exist, and — for the capability and credential
// rows — the registry must show that a remaining target can actually serve the
// request, so the chain is not walked into an identical refusal.
func (e *FallbackEngine) fallbackPermitted(ctx context.Context, run FallbackRun, idx int, class FailureClass) bool {
	if ctx.Err() != nil {
		return false
	}
	if idx+1 >= len(run.Targets) {
		return false
	}
	switch e.matrix.Lookup(class).Fallback {
	case RetryYes:
		return true
	case RetryMaybe:
		for _, next := range run.Targets[idx+1:] {
			if e.targetAdmits(ctx, run, next, class) {
				return true
			}
		}
		return false
	default:
		return false
	}
}

// targetAdmits reports whether a remaining target could plausibly serve the
// request given the failure that just occurred. Without a registry the target
// list itself is the permit, so the answer is yes.
func (e *FallbackEngine) targetAdmits(ctx context.Context, run FallbackRun, next FallbackTarget, class FailureClass) bool {
	if e.models == nil {
		return true
	}
	info, err := e.models.Resolve(ctx, next.LogicalModelID)
	if err != nil {
		return false
	}
	switch class {
	case FailureUnsupportedCapability:
		return run.RequiredCapability == "" || info.Supports(run.RequiredCapability)
	case FailureInvalidCredentials:
		return info.APIKeyEnv == "" || info.APIKey != ""
	default:
		return true
	}
}

// exhausted builds the terminal error from the trail.
func exhausted(outcome FallbackOutcome, last FallbackAttempt) error {
	return &FallbackExhaustedError{
		Attempts: outcome.Attempts,
		Chain:    outcome.Chain,
		Last:     last,
		Kind:     errorKindFor(last.Err),
	}
}

// errorKindFor maps the final error onto the transport error taxonomy,
// preserving the pre-engine behaviour: a preflight refusal (unknown model,
// missing capability, empty credential, unregistered vendor family) is a
// configuration fault, everything else is an invoke fault.
func errorKindFor(err error) ErrorKind {
	if err == nil {
		return ErrorKindInvoke
	}
	var cfgErr *ConfigError
	var capErr *CapabilityError
	var credErr *CredentialError
	var noProvider *NoProviderError
	if errors.As(err, &cfgErr) || errors.As(err, &capErr) ||
		errors.As(err, &credErr) || errors.As(err, &noProvider) {
		return ErrorKindConfig
	}
	return ErrorKindInvoke
}

// sleepWithContext waits for d, or until ctx is done.
func sleepWithContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
