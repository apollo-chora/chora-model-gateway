package domain

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"
)

// =============================================================================
// Executor — the single governed execution path for all model invocations.
//
// The Executor is the gateway's governance engine. It wraps the full per-request
// pipeline:
//
//	1. Envelope validation + invocation_id resolution.
//  2. Policy resolve (vendor + model + Armor template + fallback chain).
//  3. Keyed idempotency claim (CLAIMED).
//  4. Tenant budget load + decision (allow / downgrade / block).
//  5. Armor PRE (prompt, derived contents, derived tool declarations).
//  6. Provider dispatch with fallback-chain walk (IN_FLIGHT → SUCCEEDED).
//  7. Armor POST (completion, derived tool-call arguments).
//  8. Usage accounting: budget debit + outbox emit (ACCOUNTED).
//  9. Mana debit (optional, when the mana port is wired).
//
// The Executor is the SECURITY BOUNDARY: handlers (gRPC, HTTP) translate their
// wire requests to ExecuteRequest and call Execute; they never hold a Provider
// client and cannot reach a vendor directly. Providers are dumb adapters beneath
// the governance layer; HTTP and gRPC are transports above it.
//
// The Executor holds NO state across requests — all per-request state lives on
// the stack via the ExecuteRequest/ExecuteResponse pair. Thread-safe to call
// from multiple goroutines.
//
// NO database transaction is held open across an LLM call. The idempotency
// claim, the budget read, and the settlement (debit + outbox) each run in their
// OWN short transaction; the provider dispatch happens with no DB transaction
// in flight. This is the accounting state machine's core invariant.
// =============================================================================

// ExecuteRequest is the canonical request for all model invocations. Both the
// gRPC adapter and the HTTP API translate their wire requests to this type
// before calling Executor.Execute; the Executor never sees a proto or an HTTP
// body.
type ExecuteRequest struct {
	// Caller-supplied UUIDv7. If empty, the Executor generates one + echoes it
	// in the response. Idempotency key for ledger dedup.
	InvocationID string

	// REQUIRED — tenant scope. RLS-enforced inside BudgetRepo
	// (SET LOCAL chora.tenant_id = ...). UUIDv7.
	TenantID string

	// REQUIRED — actor identity (learner GCID or agent AGID). UUIDv7.
	GCID string

	// REQUIRED — calling agent identifier (registry.json crew name
	// or graph-node id).
	AgentID string

	// OPTIONAL — crew classification used as fallback policy key.
	CrewKind string

	// REQUIRED — logical model identifier (e.g. "gemini-2.5-pro").
	LogicalModelID LogicalModelID

	// OPTIONAL — agent-declared ordered fallback chain of logical model ids,
	// tried in order on primary dispatch failure. This is the AGENT-DRIVEN
	// tiering contract (CR qgen 2026-06-01): the calling agent owns a
	// build-time YAML declaring tier/primary/fallback, sends its primary as
	// LogicalModelID + this chain, and the gateway HONOURS it (the policy
	// loader translates these into policy.FallbackChain entries against the
	// resolved vendor). Empty = single-shot.
	FallbackModelIDs []LogicalModelID

	// REQUIRED — user-facing prompt content (screened by Armor PRE).
	Prompt string

	// OPTIONAL — output modality selector: "" (default) / "TEXT" / "IMAGE".
	// "IMAGE" routes the request to an image-capable gemini model via
	// generationConfig.responseModalities and returns PNG bytes on
	// ExecuteResponse.ImageBytes (W8, CR 2026-06-01). The tier-routing +
	// Armor + budget + fallback machinery is response-agnostic, so this
	// only changes the vendor request shape + the response carrier — not
	// the orchestration.
	ResponseModality string

	// OPTIONAL — system-instruction prompt (NOT screened — platform-controlled).
	SystemPrompt string

	// OPTIONAL — mana metering action_code (WS-1 umbrella metering, ADR-142 §4).
	// Names the priced action in chora_identity.mana_action_pricing the metering
	// seam debits per-GCID after a successful LLM call. Empty = the seam derives
	// it from AgentID / CrewKind via a fallback map (an unmapped derivation is
	// treated as un-metered + logged). For tool-calling crews (ADR-177) the
	// caller sets it ONLY on the turn-initiating call so one turn = one debit.
	ActionCode string

	// OPTIONAL — tool-calling conversation (ADR-177). JSON of the genai
	// []*Content multi-turn history. When non-empty the vendor adapter builds the
	// model request from this and ignores Prompt. Empty = legacy text-only path.
	ContentsJSON string

	// OPTIONAL — tool/function declarations (ADR-177). JSON of the genai []*Tool
	// the model may call. Empty = plain generation.
	ToolsJSON string

	// OPTIONAL — vendor-neutral generation parameters.
	GenerationConfig map[string]any

	// REQUIRED for distributed tracing — W3C traceparent.
	Traceparent string

	// OPTIONAL — W3C tracestate.
	Tracestate string

	// REQUIRED (ADR-254 D7 / ADR-252 Q1): the crew id of the invoking agent
	// (companion_chat | companion_diagnosis | kg_exploration | qgen | oe_grading |
	// content_recommender | content_moderation | duel_atom_smith |
	// profile_conjurer). The caller's honest declaration of what surface the
	// call is; the companion-suspension gate refuses an ABSENT surface
	// loudly (surface_unstamped) before any debit, and reads the containment
	// tables only for the companion surfaces.
	Surface string

	// OPTIONAL (ADR-254 D7 / R22): the dispatch idempotency_key the agent
	// forwards from its Pub/Sub request. When set, the metering seam takes a
	// keyed debit claim on (gcid, key, action_code) before the mana debit, so a
	// redelivered dispatch bills once. Empty on non-dispatched calls.
	DispatchIdempotencyKey string
}

// ExecuteResponse is the canonical response for all model invocations. Both
// the gRPC adapter and the HTTP API translate FROM this type to their wire
// responses; the Executor never builds a proto or an HTTP body.
type ExecuteResponse struct {
	InvocationID string
	Completion   string
	// ImageBytes carries the generated image (e.g. PNG) for image-modality
	// responses; empty for text. ImageMIMEType is the corresponding MIME
	// (e.g. "image/png"). Both populated only when the resolved model
	// returned an inline image (W8, CR 2026-06-01).
	ImageBytes    []byte
	ImageMIMEType string
	Usage         TokenUsage
	Vendor        string // resolved vendor identifier (post-fallback)
	ModelVersion  string // resolved model version (post-optimizer)
	ArmorPre      ArmorVerdict
	ArmorPost     ArmorVerdict
	// ArmorPreContents is the verdict on the text derived from `contents_json`
	// (G1'-3). Bypassed when the tier is permissive or the turn carries no
	// screenable contents; Unspecified when the screen itself faulted, which
	// is deliberately NOT reported as Allow.
	ArmorPreContents ArmorVerdict
	// ArmorPostToolCalls is the verdict on the text derived from
	// `tool_calls_json` (G1'-2). Bypassed when the tier is permissive or the
	// turn emitted no screenable tool arguments; Unspecified when the screen
	// itself faulted, which is deliberately NOT reported as Allow.
	ArmorPostToolCalls ArmorVerdict
	// ArmorPreTools is the verdict on the text derived from `tools_json`
	// (CHO-2391): the caller-supplied tool names and descriptions. Bypassed
	// when the tier is permissive or the turn declares no screenable tools;
	// Unspecified when the screen itself faulted, which is deliberately NOT
	// reported as Allow.
	ArmorPreTools  ArmorVerdict
	FallbackChain  []string // one entry per attempted vendor:model
	LatencyMs      int32
	FinishReason   FinishReason
	FinishDetail   string
	CompletedAt    time.Time
	GatewayVersion string // e.g. "chora-model-gateway:58b28eb"

	// ToolCallsJSON carries the model-emitted function calls this turn (ADR-177).
	// Non-empty ⇒ the agent must execute the tools + continue the loop; empty ⇒
	// terminal text turn. JSON of genai functionCall parts.
	ToolCallsJSON string
	// RevisedPrompt carries the provider-revised prompt for image generation
	// (OpenAI images/generations revised_prompt field). Empty for text turns
	// and for vendors that do not revise.
	RevisedPrompt string
	// Citations are the sources the model cited in its completion (vendor-
	// parsed). Empty when the vendor returned none.
	Citations []Citation
	// SearchQueries are the web-search queries the model issued on a grounded
	// turn. Empty on ungrounded turns.
	SearchQueries []string

	// Deduped reports whether this invocation was a redelivery of a dispatch
	// idempotency key that an earlier delivery already claimed. The Executor
	// sets it from the keyed debit claim; the mana-metering decorator reads it
	// to skip the mana debit on a redelivered dispatch (one claim suppresses
	// both the tenant-budget debit and the mana debit).
	Deduped bool

	// AccountingState is the final state of the per-invocation accounting
	// lifecycle the Executor drove (see AccountingState). It is observability
	// only — the settlement outcome is already reflected in Usage + the outbox
	// event — but it lets a caller tell "the provider succeeded but settlement
	// failed" (SUCCEEDED) from "the provider never ran" (CLAIMED) without
	// correlating logs.
	AccountingState AccountingState
}

// Provider is the vendor-dispatch port the Executor drives. It is the ONLY way
// a model call reaches a vendor: the Executor holds the Provider registry and
// no handler (gRPC, HTTP) can construct or reach a Provider directly.
//
// The interface is structurally identical to VendorClient (ports.go) — every
// VendorClient implementation satisfies Provider — but it is declared
// separately so the Executor's dependency is explicit: the governance layer
// depends on Provider, never on a concrete vendor adapter.
type Provider interface {
	// Family returns the VendorFamily this provider implements. The Executor
	// uses it to pick the right provider from the registry at dispatch time.
	Family() VendorFamily

	// Generate executes the LLM call. Returns the completion + token usage
	// + finish reason. Errors are returned verbatim — the Executor decides
	// whether to walk the fallback chain.
	Generate(ctx context.Context, req VendorRequest) (VendorResponse, error)
}

// AccountingState is the per-invocation accounting lifecycle the Executor
// drives. It makes the settlement flow explicit and auditable: a call is
// CLAIMED (idempotency key taken), IN_FLIGHT (provider dispatch running),
// SUCCEEDED (provider returned), ACCOUNTED (budget debited + outbox emitted).
//
// The transitions are one-way and monotonic. A terminal failure leaves the
// state at the last successful transition (or FAILED), so a consumer can tell
// exactly how far the call got: a response with SUCCEEDED but no ledger row
// means the provider ran and settlement failed; a response with CLAIMED means
// the call never reached a provider.
type AccountingState int

const (
	// AccountingUnspecified is the zero value — no accounting step ran.
	AccountingUnspecified AccountingState = iota
	// AccountingClaimed — the keyed idempotency claim was taken (or there was
	// no dispatch key to claim). The call is eligible to spend.
	AccountingClaimed
	// AccountingInFlight — the provider dispatch has begun. No DB transaction
	// is held across this window (the claim's transaction already committed).
	AccountingInFlight
	// AccountingSucceeded — the provider returned a completion. The call is
	// billable; settlement (debit + outbox) has NOT yet run.
	AccountingSucceeded
	// AccountingAccounted — the budget debit + outbox emit committed. The
	// call is fully settled; this is the terminal happy-path state.
	AccountingAccounted
	// AccountingFailed — a terminal failure occurred. The state records the
	// last successful transition so a consumer can see how far the call got.
	AccountingFailed
)

// String returns a human-readable name for logs + span attributes.
func (s AccountingState) String() string {
	switch s {
	case AccountingClaimed:
		return "claimed"
	case AccountingInFlight:
		return "in_flight"
	case AccountingSucceeded:
		return "succeeded"
	case AccountingAccounted:
		return "accounted"
	case AccountingFailed:
		return "failed"
	default:
		return "unspecified"
	}
}

// SuspensionGate is the optional containment-check port for the Executor. When
// wired, the Executor refuses a contained or unstamped turn BEFORE any spend
// (deny-before-debit), mirroring the ADR-252 containment decorator. When nil,
// the Executor skips the suspension check — the transport-level middleware
// (adapter/middleware.CompanionSuspension) is then the gate, and the Executor
// must not double-gate.
type SuspensionGate interface {
	// Check returns whether the turn is permitted. A non-nil error means the
	// containment state could not be read; the Executor fails CLOSED (refuses)
	// because an unreadable gate must not allow a contained turn through.
	Check(ctx context.Context, req ExecuteRequest) (SuspensionVerdict, error)
}

// SuspensionVerdict is the result of one containment check.
type SuspensionVerdict struct {
	// Suspended is true when the turn must be refused.
	Suspended bool
	// Reason is the machine token (DenySurfaceUnstamped,
	// DenyCompanionSuspended, DenySuspensionUnreadable) when Suspended.
	Reason string
	// Detail is the human-readable explanation.
	Detail string
}

// Executor wraps the governance pipeline for all model invocations. It is the
// single entry point (Execute) that both the gRPC and HTTP transports call.
//
// The Executor holds NO state across requests. All per-request state lives on
// the stack via the ExecuteRequest/ExecuteResponse pair.
type Executor struct {
	// fallback owns the cross-provider walk: the ordered target list, the
	// per-target preflight, the retry matrix, and the rule that only the
	// winning attempt is billable. The Executor never loops over providers
	// itself. The engine holds no database port, so no transaction can be
	// held across an LLM call by construction.
	fallback *FallbackEngine

	armor    ArmorClient
	budget   BudgetRepo
	outbox   OutboxWriter
	policies PolicyLoader

	// claims is the keyed debit-claim port (ADR-254 D7, R22). OPTIONAL at
	// construction; when nil the Execute flow skips the claim (no dispatch
	// dedup). Production (cmd/server/main.go) wires the pgrepo.
	claims DebitClaimer

	// atomicOutbox is the transactional debit+outbox port. OPTIONAL at
	// construction; when nil the Executor falls back to separate
	// DebitSpent + EnqueueTokenUsageRecorded calls. Production wires the
	// pg repo (which implements AtomicBudgetOutbox).
	atomicOutbox AtomicBudgetOutbox

	// violations is the ADR-152 governance producer (amendment 2026-08-07).
	// OPTIONAL at construction. When a BLOCK happens with no publisher wired
	// the Executor logs at ERROR rather than dropping the evidence silently.
	violations ViolationPublisher

	// suspension is the optional containment gate (ADR-252). OPTIONAL at
	// construction; when nil the Executor skips the suspension check (the
	// transport-level middleware is the gate). When wired, the Executor
	// refuses a contained turn BEFORE any spend.
	suspension SuspensionGate

	// mana is the optional per-GCID mana meter (WS-1 umbrella metering,
	// ADR-142 §4). OPTIONAL at construction; when nil the Executor skips the
	// mana debit (the transport-level middleware is the gate). When wired,
	// the Executor debits post-success, idempotent on the invocation id.
	mana ManaMeter

	// dbTimeout bounds each database operation (budget read, idempotency
	// claim, settlement debit + outbox). A slow DB must not consume the
	// entire request budget — the provider dispatch is the expensive part
	// and needs the remaining time. Zero means no DB timeout (the request
	// context is the only bound).
	dbTimeout time.Duration

	gatewayVersion string // build identifier, e.g. "chora-model-gateway:58b28eb"
	now            func() time.Time
	newID          func() string // UUIDv7 generator for invocation_id default

	// defaultAgentID is the fallback agent id for IMAGE action_code
	// defaulting: when modality is IMAGE and the caller stamped no
	// action_code, the Executor derives "{defaultAgentID}_image".
	defaultAgentID string

	// enforceContentsScreen promotes the G1'-3 contents leg to blocking.
	// See ExecutorConfig.EnforceContentsScreen.
	enforceContentsScreen bool

	// enforceToolCallScreen promotes the G1'-2 tool-call leg to blocking.
	// See ExecutorConfig.EnforceToolCallScreen.
	enforceToolCallScreen bool

	// enforceToolsScreen promotes the CHO-2391 tool-DECLARATION leg to
	// blocking. See ExecutorConfig.EnforceToolsScreen.
	enforceToolsScreen bool
}

// ExecutorConfig groups the ports + cross-cutting deps the Executor needs.
type ExecutorConfig struct {
	// Providers is one Provider per VendorFamily. REQUIRED — at least one.
	Providers []Provider
	Armor     ArmorClient
	Budget    BudgetRepo
	Outbox    OutboxWriter
	Policies  PolicyLoader

	// Claims is the keyed debit-claim port (ADR-254 D7, R22). OPTIONAL; nil
	// means Execute skips the dispatch dedup claim.
	Claims DebitClaimer

	// Models is the model-registry resolver. OPTIONAL; nil means the
	// capability / output-ceiling / credential checks are skipped.
	Models ModelResolver

	// FallbackRetries is the number of ADDITIONAL same-provider attempts the
	// FallbackEngine may make after a retryable failure (timeout / 429 / 5xx).
	// 0 — the default — disables same-provider retry: the walk advances
	// straight to the next target, so the matrix's "maybe" rows read as "no"
	// until an operator opts in. Sourced from CHORA_FALLBACK_RETRIES.
	FallbackRetries int

	// FallbackRetryBackoff is the delay before each same-provider retry.
	// Sourced from CHORA_FALLBACK_RETRY_BACKOFF_MS.
	FallbackRetryBackoff time.Duration

	// RetryMatrix overrides the ratified retry table (DefaultRetryMatrix).
	// OPTIONAL; the zero value keeps the default. Tests use it to prove a
	// caller cannot re-enable fallback on a policy refusal.
	RetryMatrix RetryMatrix

	// Sleep is the backoff sleeper the FallbackEngine uses. OPTIONAL; nil uses
	// a context-aware timer. Tests substitute a no-op.
	Sleep func(ctx context.Context, d time.Duration) error

	// AtomicOutbox is the transactional debit+outbox port. OPTIONAL; nil
	// means the Executor falls back to separate DebitSpent + outbox calls.
	AtomicOutbox AtomicBudgetOutbox

	// Violations is the ADR-152 governance-violation producer. OPTIONAL, see
	// the field comment on Executor.violations.
	Violations ViolationPublisher

	// Suspension is the optional containment gate (ADR-252). OPTIONAL; nil
	// means the Executor skips the suspension check.
	Suspension SuspensionGate

	// Mana is the optional per-GCID mana meter. OPTIONAL; nil means the
	// Executor skips the mana debit.
	Mana ManaMeter

	GatewayVersion string

	// Optional overrides for deterministic tests. Production wiring leaves
	// these nil and the constructor substitutes time.Now + UUIDv7.
	Now   func() time.Time
	NewID func() string

	// DefaultAgentID is the fallback agent id for IMAGE action_code
	// defaulting. Empty means no IMAGE defaulting.
	DefaultAgentID string

	// EnforceContentsScreen promotes the G1'-3 contents leg from audit-only to
	// blocking. Default false, which is the owner's closing rule of 2026-08-07:
	// a newly screened path must not start refusing calls that pass today. When
	// false the verdict is still computed, reported and published as governance
	// evidence, so the promotion decision can be made on measured data rather
	// than on expectation. Sourced from CHORA_ARMOR_ENFORCE_CONTENTS_SCREEN.
	EnforceContentsScreen bool

	// EnforceToolCallScreen promotes the G1'-2 tool-call leg from audit-only to
	// blocking, on exactly the same terms as EnforceContentsScreen above:
	// tool-calling turns succeed today, so by default the verdict is computed,
	// reported and published as evidence while the turn proceeds. Sourced from
	// CHORA_ARMOR_ENFORCE_TOOL_CALL_SCREEN.
	EnforceToolCallScreen bool

	// EnforceToolsScreen promotes the CHO-2391 tool-DECLARATION leg from
	// audit-only to blocking, on exactly the same terms as the two above.
	// Default false is load-bearing here: tool declarations are
	// platform-authored today, so every tool-calling turn in production passes
	// this path, and a leg that starts refusing them would break working
	// callers to close a hole nothing currently reaches. Sourced from
	// CHORA_ARMOR_ENFORCE_TOOLS_SCREEN.
	EnforceToolsScreen bool

	// DBTimeout bounds each database operation (budget read, idempotency
	// claim, settlement debit + outbox). A slow DB must not consume the
	// entire request budget — the provider dispatch is the expensive part
	// and needs the remaining time. Zero means no DB timeout (the request
	// context is the only bound).
	DBTimeout time.Duration
}

// NewExecutor constructs an Executor. Returns an error if any required port
// is nil — fail-loud per feedback_no_stubs_real_wiring (no silent fallback
// to a stub adapter).
func NewExecutor(cfg ExecutorConfig) (*Executor, error) {
	if cfg.Armor == nil {
		return nil, errors.New("model gateway: ArmorClient required")
	}
	if cfg.Budget == nil {
		return nil, errors.New("model gateway: BudgetRepo required")
	}
	if cfg.Outbox == nil {
		return nil, errors.New("model gateway: OutboxWriter required")
	}
	if cfg.Policies == nil {
		return nil, errors.New("model gateway: PolicyLoader required")
	}
	if len(cfg.Providers) == 0 {
		return nil, errors.New("model gateway: at least one Provider required")
	}
	if cfg.GatewayVersion == "" {
		return nil, errors.New("model gateway: GatewayVersion required (no inline-config-style empty defaults)")
	}

	providers := make(map[VendorFamily]Provider, len(cfg.Providers))
	for _, p := range cfg.Providers {
		if p == nil {
			return nil, errors.New("model gateway: nil Provider in Providers slice")
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

	// The FallbackEngine owns the cross-provider walk. It is built from the
	// SAME providers + model resolver the Executor was given, so there is one
	// dispatch registry, not two.
	fallback, err := NewFallbackEngine(FallbackConfig{
		Providers:              cfg.Providers,
		Models:                 cfg.Models,
		Matrix:                 cfg.RetryMatrix,
		MaxSameProviderRetries: cfg.FallbackRetries,
		RetryBackoff:           cfg.FallbackRetryBackoff,
		Sleep:                  cfg.Sleep,
	})
	if err != nil {
		return nil, err
	}

	now := cfg.Now
	if now == nil {
		now = time.Now
	}
	newID := cfg.NewID
	if newID == nil {
		newID = func() string {
			// Production wiring substitutes a real UUIDv7 generator in
			// cmd/server/main.go. The default here intentionally panics so
			// a no-op-default-substitution mistake fails loud at test time.
			panic("model gateway: NewID generator not wired — set ExecutorConfig.NewID")
		}
	}

	return &Executor{
		fallback:       fallback,
		armor:          cfg.Armor,
		budget:         cfg.Budget,
		outbox:         cfg.Outbox,
		policies:       cfg.Policies,
		claims:         cfg.Claims,
		atomicOutbox:   cfg.AtomicOutbox,
		violations:     cfg.Violations,
		suspension:     cfg.Suspension,
		mana:           cfg.Mana,
		gatewayVersion: cfg.GatewayVersion,
		now:            now,
		newID:          newID,
		defaultAgentID: cfg.DefaultAgentID,

		enforceContentsScreen: cfg.EnforceContentsScreen,
		enforceToolCallScreen: cfg.EnforceToolCallScreen,
		enforceToolsScreen:    cfg.EnforceToolsScreen,
	}, nil
}

// Execute is the single governed entry point for all model invocations. Both
// the gRPC adapter and the HTTP API translate their wire requests to
// ExecuteRequest and call this method; the response translates back at the
// wire boundary.
//
// Order of operations (each step is one port call — easy to trace):
//
//  1. Validate the envelope + resolve invocation_id.
//  2. Suspension check (OPTIONAL — only when the suspension port is wired).
//  3. PolicyLoader.ResolveAgentPolicy — pick vendor + model + Armor template.
//  4. Keyed idempotency claim (→ CLAIMED).
//  5. BudgetRepo.GetTenantBudget + Decide — allow / downgrade / block.
//  6. ArmorClient.SanitizeUserPrompt (PRE) on the prompt, the derived
//     contents text, and the derived tool declarations.
//  7. Provider dispatch with fallback-chain walk (→ IN_FLIGHT → SUCCEEDED).
//  8. ArmorClient.SanitizeModelResponse (POST) on the completion + the derived
//     tool-call arguments.
//  9. Usage accounting: budget debit + outbox emit (→ ACCOUNTED). NO DB
//     transaction is held across the LLM call — the claim's transaction
//     committed before dispatch and this settlement runs in its OWN short
//     transaction.
//  10. Mana debit (OPTIONAL — only when the mana port is wired).
//
// The accounting state machine (CLAIMED → IN_FLIGHT → SUCCEEDED → ACCOUNTED)
// is tracked throughout and reported on ExecuteResponse.AccountingState.
func (e *Executor) Execute(ctx context.Context, req ExecuteRequest) (*ExecuteResponse, error) {
	start := e.now()
	acctState := AccountingUnspecified

	// Step 1 — validate the envelope. A missing tenant or model is a caller
	// bug, and discovering it at the vendor would mean a billable request
	// with no attribution. Modality must be one of the known tokens; an
	// empty request (no prompt AND no contents) is still billed by most
	// providers, so it is refused here rather than at the vendor.
	switch {
	case req.TenantID == "":
		return nil, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "tenant_id required",
		}
	case req.GCID == "":
		return nil, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "gcid required",
		}
	case req.AgentID == "":
		return nil, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "agent_id required",
		}
	case req.LogicalModelID == "":
		return nil, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "logical_model_id required",
		}
	}
	switch req.ResponseModality {
	case "", "TEXT", "IMAGE", "GROUNDED":
		// known modalities
	default:
		return nil, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: fmt.Sprintf("response_modality %q is not one of TEXT, IMAGE, GROUNDED", req.ResponseModality),
		}
	}
	if req.Prompt == "" && req.ContentsJSON == "" {
		return nil, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "prompt or contents_json required (an empty request is still billed by most providers)",
		}
	}

	// Step 1b — resolve invocation_id.
	invocationID := req.InvocationID
	if invocationID == "" {
		invocationID = e.newID()
	}

	// IMAGE action_code defaulting: when the modality is IMAGE and the caller
	// stamped no action_code, derive "{defaultAgentID}_image" so the image
	// call is metered under a distinct action. Derived ONCE here so the
	// idempotency claim, the mana debit and the outbox event all carry the
	// same code (Python parity: `request.get("action_code") or agent`).
	actionCode := req.ActionCode
	if req.ResponseModality == "IMAGE" && actionCode == "" && e.defaultAgentID != "" {
		actionCode = e.defaultAgentID + "_image"
	}
	if actionCode == "" {
		actionCode = req.AgentID
	}

	// Step 2 — suspension check (OPTIONAL). When the suspension port is wired
	// the Executor is the containment gate; when nil the transport-level
	// middleware (adapter/middleware.CompanionSuspension) is the gate and the
	// Executor must not double-gate. Fail CLOSED on a read error: an
	// unreadable gate must not allow a contained turn through (ADR-252 D6).
	if e.suspension != nil {
		v, serr := e.suspension.Check(ctx, req)
		switch {
		case serr != nil:
			return nil, &PreconditionError{
				Reason: DenySuspensionUnreadable,
				Detail: "companion containment state unreadable; refusing (fail closed)",
				Inner:  serr,
			}
		case v.Suspended:
			return nil, &PreconditionError{
				Reason: v.Reason,
				Detail: v.Detail,
			}
		}
	}

	// Step 3 — resolve agent policy. The agent-declared fallback chain
	// (req.FallbackModelIDs) flows into the loader so it can populate
	// policy.FallbackChain against the resolved vendor (AGENT-DRIVEN tiering,
	// CR qgen 2026-06-01).
	policy, err := e.policies.ResolveAgentPolicy(ctx, req.AgentID, req.CrewKind, req.LogicalModelID, req.FallbackModelIDs)
	if err != nil {
		return nil, &InvokeError{
			Reason: FinishReasonVendorError, // policy resolve failure surfaces as upstream-like error
			Detail: "policy resolve failed",
			Inner:  err,
		}
	}

	// Step 4 — keyed idempotency claim (→ CLAIMED). A redelivered dispatch
	// with the same key must bill once, so the claim is taken BEFORE any
	// spend. The claim result suppresses BOTH the tenant-budget debit
	// (step 9) and the mana debit (the middleware reads
	// ExecuteResponse.Deduped). A claim failure is a hard error: proceeding
	// would risk a double bill.
	//
	// The claim runs in its OWN short transaction (inside the DebitClaimer
	// implementation) which commits BEFORE the provider dispatch begins —
	// no DB transaction is held across the LLM call.
	deduped := false
	if key := strings.TrimSpace(req.DispatchIdempotencyKey); key != "" && e.claims != nil {
		dbCtx, dbCancel := e.dbContext(ctx)
		claimed, cerr := e.claims.ClaimDebit(dbCtx, req.GCID, key, actionCode, invocationID)
		dbCancel()
		switch {
		case cerr != nil:
			return nil, &InvokeError{
				Reason: FinishReasonVendorError,
				Detail: fmt.Sprintf("idempotency claim failed: %v", cerr),
				Inner:  cerr,
			}
		case !claimed:
			deduped = true
		}
	}
	acctState = AccountingClaimed

	// Step 5 — load tenant budget.
	dbCtx, dbCancel := e.dbContext(ctx)
	budget, err := e.budget.GetTenantBudget(dbCtx, req.TenantID)
	dbCancel()
	if err != nil {
		return nil, &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "budget repo unavailable",
			Inner:  err,
		}
	}

	// Step 6 — decide budget action.
	decision := BudgetAllow
	if budget != nil {
		decision = budget.Decide()
		if decision == BudgetDowngrade && budget.DowngradeToModel != "" {
			policy.ResolvedLogicalModelID = budget.DowngradeToModel
		}
	}
	if decision == BudgetBlock {
		return &ExecuteResponse{
			InvocationID:    invocationID,
			ArmorPre:        ArmorVerdictUnspecified, // never reached
			ArmorPost:       ArmorVerdictUnspecified,
			FallbackChain:   []string{},
			LatencyMs:       int32(e.now().Sub(start).Milliseconds()), // #nosec G115 -- elapsed ms since request start; bounded far below int32 (overflows only past ~24.8 days)
			FinishReason:    FinishReasonBudgetBlock,
			FinishDetail:    "tenant LLM budget exhausted; policy=block",
			CompletedAt:     e.now(),
			GatewayVersion:  e.gatewayVersion,
			AccountingState: acctState,
		}, nil
	}

	// Step 6 — Armor PRE on the user prompt.
	armorPre := ArmorVerdictBypassed
	sanitisedPrompt := req.Prompt
	if policy.ArmorTemplate != "" {
		dbCtx, dbCancel := e.dbContext(ctx)
		armorPre, sanitisedPrompt, err = e.armor.SanitizeUserPrompt(dbCtx, policy.ArmorTemplate, req.Prompt)
		dbCancel()
		if err != nil {
			return nil, &InvokeError{
				Reason: FinishReasonVendorError,
				Detail: "model armor PRE call failed",
				Inner:  err,
			}
		}
		if armorPre.IsTerminalBlock() {
			// Emit a partial-cost ledger event covering input-only. Outbox
			// errors here are accepted at the domain layer (the block
			// response is the user-visible outcome and is more important
			// than ledger durability on this path) but the OutboxWriter
			// implementation MUST record the failure on the OTel span via
			// its own instrumentation. Explicit `_ =` satisfies errcheck
			// without changing this contract.
			_ = e.emitOutbox(ctx, invocationID, req, policy, TokenUsage{}, armorPre, ArmorVerdictUnspecified, []string{}, "")
			e.emitPolicyViolation(ctx, invocationID, req, policy, ArmorLegPre, armorPre)
			return &ExecuteResponse{
				InvocationID:    invocationID,
				ArmorPre:        armorPre,
				ArmorPost:       ArmorVerdictUnspecified,
				FallbackChain:   []string{},
				LatencyMs:       int32(e.now().Sub(start).Milliseconds()), // #nosec G115 -- elapsed ms since request start; bounded far below int32 (overflows only past ~24.8 days)
				FinishReason:    FinishReasonModelArmorBlock,
				FinishDetail:    "model armor PRE blocked the prompt",
				CompletedAt:     e.now(),
				GatewayVersion:  e.gatewayVersion,
				AccountingState: acctState,
			}, nil
		}
	}

	// Step 6b. Armor PRE on the text DERIVED from contents_json (G1'-3).
	//
	// Step 6 screens req.Prompt. On the multimodal path the caller's prompt is
	// a fixed placeholder and the learner's uploaded artifact rides in
	// contents_json, so step 6 screens a constant and the artifact reaches the
	// model unscreened. Screening here rather than fixing the one caller means
	// no future caller can re-open the hole by getting the contract wrong.
	//
	// POSTURE, owner's closing rule 2026-08-07: this screens content that has
	// never been screened, so it is AUDIT-ONLY by default. The verdict is
	// computed, returned and published as governance evidence, and the turn
	// proceeds. Nothing that passes today starts being refused. Promotion is
	// EnforceContentsScreen, a config flip, once there is measured data.
	//
	// A permissive tier (empty template) adds no leg here, exactly as it adds
	// none at step 6.
	armorPreContents := ArmorVerdictBypassed
	if policy.ArmorTemplate != "" {
		if derived := DeriveScreenableText(req.ContentsJSON); derived != "" {
			verdict, _, cErr := e.armor.SanitizeUserPrompt(ctx, policy.ArmorTemplate, derived)
			switch {
			case cErr != nil:
				// A fault is not a verdict, and must never be recorded as
				// Allow. While the leg is audit-only a screening outage
				// degrades to unscreened rather than refusing a learner's
				// upload; once enforcing, it fails loud like any other
				// guardrail fault.
				if e.enforceContentsScreen {
					return nil, &InvokeError{
						Reason: FinishReasonVendorError,
						Detail: "model armor PRE (contents) call failed",
						Inner:  cErr,
					}
				}
				armorPreContents = ArmorVerdictUnspecified
				slog.ErrorContext(ctx, "model gateway: contents screen faulted, turn proceeded UNSCREENED (audit-only leg)",
					"agent_id", req.AgentID, "tenant_id", req.TenantID,
					"invocation_id", invocationID, "error", cErr)
			default:
				armorPreContents = verdict
				if verdict.IsTerminalBlock() {
					// Evidence is emitted in BOTH postures. The whole purpose
					// of the audit-only phase is to produce the measurement
					// that justifies promoting it.
					e.emitPolicyViolation(ctx, invocationID, req, policy, ArmorLegPreContents, verdict)
					slog.WarnContext(ctx, "model gateway: contents screen BLOCKED",
						"agent_id", req.AgentID, "tenant_id", req.TenantID,
						"invocation_id", invocationID, "verdict", verdict.String(),
						"enforced", e.enforceContentsScreen,
						"posture", map[bool]string{true: "refused", false: "audit-only, turn proceeded"}[e.enforceContentsScreen])
					if e.enforceContentsScreen {
						_ = e.emitOutbox(ctx, invocationID, req, policy, TokenUsage{}, armorPre, ArmorVerdictUnspecified, []string{}, "")
						return &ExecuteResponse{
							InvocationID:     invocationID,
							ArmorPre:         armorPre,
							ArmorPreContents: armorPreContents,
							ArmorPost:        ArmorVerdictUnspecified,
							FallbackChain:    []string{},
							LatencyMs:        int32(e.now().Sub(start).Milliseconds()), // #nosec G115 -- elapsed ms since request start; bounded far below int32
							FinishReason:     FinishReasonModelArmorBlock,
							FinishDetail:     "model armor PRE blocked the request contents",
							CompletedAt:      e.now(),
							GatewayVersion:   e.gatewayVersion,
							AccountingState:  acctState,
						}, nil
					}
				}
			}
		}
	}

	// Step 6c. Armor PRE on the text DERIVED from tools_json (CHO-2391).
	//
	// Steps 6 and 6b screen the prompt and the conversation. NEITHER reads tool
	// DECLARATIONS, so `tools_json` arrives from the caller screened by nothing.
	// Demonstrated live on the deployed build during G1'-2 acceptance: the
	// prompt was benign and passed PRE with ALLOW, the injection rode in a tool
	// DESCRIPTION, and the model copied it verbatim into a tool argument. The
	// POST_TOOL_CALLS leg caught it on the way OUT, which is the only net today
	// and is a net that only closes after the model has already been steered.
	//
	// Not an incident today, because tool declarations are platform-authored
	// and nothing learner-reachable supplies them. That bound disappears the
	// moment a tenant-supplied or agent-generated tool set reaches the gateway,
	// BYOA being the obvious candidate. Screening HERE rather than trusting the
	// caller is what stops a future caller re-opening it.
	//
	// POSTURE, owner's closing rule 2026-08-07 and the precedent steps 6b and
	// POST_TOOL_CALLS set: this screens content that has never been screened,
	// so it is AUDIT-ONLY by default. The default matters more here than on the
	// other two legs, because EVERY tool-calling turn in production carries
	// platform-authored declarations through this path: an enforcing default
	// would refuse working callers to close a hole nothing currently reaches.
	// Promotion is EnforceToolsScreen, a config flip, once there is measured
	// data.
	//
	// A permissive tier (empty template) adds no leg here, exactly as at step 6.
	armorPreTools := ArmorVerdictBypassed
	if policy.ArmorTemplate != "" {
		if derived := DeriveScreenableToolText(req.ToolsJSON); derived != "" {
			verdict, _, tErr := e.armor.SanitizeUserPrompt(ctx, policy.ArmorTemplate, derived)
			switch {
			case tErr != nil:
				// A fault is not a verdict, and must never be recorded as
				// Allow. While the leg is audit-only a screening outage
				// degrades to unscreened rather than refusing the turn; once
				// enforcing, it fails loud like any other guardrail fault.
				if e.enforceToolsScreen {
					return nil, &InvokeError{
						Reason: FinishReasonVendorError,
						Detail: "model armor PRE (tools) call failed",
						Inner:  tErr,
					}
				}
				armorPreTools = ArmorVerdictUnspecified
				slog.ErrorContext(ctx, "model gateway: tools screen faulted, turn proceeded UNSCREENED (audit-only leg)",
					"agent_id", req.AgentID, "tenant_id", req.TenantID,
					"invocation_id", invocationID, "error", tErr)
			default:
				armorPreTools = verdict
				if verdict.IsTerminalBlock() {
					// Evidence is emitted in BOTH postures. The whole purpose
					// of the audit-only phase is to produce the measurement
					// that justifies promoting it.
					e.emitPolicyViolation(ctx, invocationID, req, policy, ArmorLegPreTools, verdict)
					slog.WarnContext(ctx, "model gateway: tools screen BLOCKED",
						"agent_id", req.AgentID, "tenant_id", req.TenantID,
						"invocation_id", invocationID, "verdict", verdict.String(),
						"enforced", e.enforceToolsScreen,
						"posture", map[bool]string{true: "refused", false: "audit-only, turn proceeded"}[e.enforceToolsScreen])
					if e.enforceToolsScreen {
						_ = e.emitOutbox(ctx, invocationID, req, policy, TokenUsage{}, armorPre, ArmorVerdictUnspecified, []string{}, "")
						return &ExecuteResponse{
							InvocationID:     invocationID,
							ArmorPre:         armorPre,
							ArmorPreContents: armorPreContents,
							ArmorPreTools:    armorPreTools,
							ArmorPost:        ArmorVerdictUnspecified,
							FallbackChain:    []string{},
							LatencyMs:        int32(e.now().Sub(start).Milliseconds()), // #nosec G115 -- elapsed ms since request start; bounded far below int32
							FinishReason:     FinishReasonModelArmorBlock,
							FinishDetail:     "model armor PRE blocked the tool declarations",
							CompletedAt:      e.now(),
							GatewayVersion:   e.gatewayVersion,
							AccountingState:  acctState,
						}, nil
					}
				}
			}
		}
	}

	// Step 7 — primary provider dispatch with fallback-chain walk.
	//
	// The FallbackEngine owns the walk: the ordered target list, the per-target
	// preflight (capability, credential, output-ceiling re-clamp), the explicit
	// retry matrix, and the accounting rule that ONLY the winning attempt is
	// billable. The Executor supplies the targets + the canonical vendor
	// request and consumes the outcome — it never loops over providers itself.
	//
	// The fallback_chain slice on ExecuteResponse + the outbox event records
	// EVERY attempted vendor:model pair so consumers can attribute cost +
	// latency to the real (vs intended) dispatch.
	//
	// A policy refusal never reaches this step: Armor PRE returns at step 6 and
	// Armor POST returns at step 8, and the matrix gives both the
	// pre/post_policy_rejection rows a Fallback of "no". Asking a second
	// provider to produce a different answer to a question the platform already
	// decided is a policy choice, never an automatic recovery.
	//
	// → IN_FLIGHT: the provider dispatch begins here. NO DB transaction is held
	// across this window — the claim's transaction (step 4) already committed
	// and the settlement transaction (step 9) has not begun. The FallbackEngine
	// has no database port, so it CANNOT hold one.
	acctState = AccountingInFlight

	// IMAGE action_code defaulting: when the modality is IMAGE and the caller
	// stamped no action_code, derive "{defaultAgentID}_image" so the image
	// call is metered under a distinct action. The derived code is written
	// back onto req so the outbox event carries it.
	if req.ResponseModality == "IMAGE" && req.ActionCode == "" && e.defaultAgentID != "" {
		req.ActionCode = e.defaultAgentID + "_image"
	}

	requiredCapability := "chat"
	if req.ResponseModality == "IMAGE" {
		requiredCapability = "image"
	} else if req.ResponseModality == "GROUNDED" {
		requiredCapability = "web_search"
	}

	outcome, dispatchErr := e.fallback.Run(ctx, FallbackRun{
		Targets: e.fallback.Targets(
			AgentPolicyFallback{Vendor: policy.Vendor, ResolvedLogicalModelID: policy.ResolvedLogicalModelID},
			policy.FallbackChain,
		),
		Base: VendorRequest{
			Prompt:           sanitisedPrompt,
			ResponseModality: req.ResponseModality,
			SystemPrompt:     req.SystemPrompt,
			ContentsJSON:     req.ContentsJSON,
			ToolsJSON:        req.ToolsJSON,
			Traceparent:      req.Traceparent,
			Tracestate:       req.Tracestate,
			TenantID:         req.TenantID,
		},
		GenerationConfig:   req.GenerationConfig,
		RequiredCapability: requiredCapability,
	})
	if dispatchErr != nil {
		return nil, e.fallbackInvokeError(dispatchErr)
	}
	vendorResp := outcome.Response
	fallbackChain := outcome.Chain
	// Replace primary's resolved vendor + model with the one that actually
	// succeeded so ExecuteResponse + outbox carry the truth.
	policy.Vendor = outcome.Target.Vendor
	policy.ResolvedLogicalModelID = outcome.Target.LogicalModelID

	// → SUCCEEDED: the provider returned a completion. The call is billable;
	// settlement (step 9) has NOT yet run. Only the winning attempt's usage is
	// carried forward — a failed attempt's partial output is not billable.
	acctState = AccountingSucceeded

	// Step 8 — Armor POST on the completion.
	//
	// Cloud Model Armor is a TEXT guardrail: it screens the completion string.
	// For an IMAGE-only response (no completion text, image bytes present)
	// there is nothing for the text-sanitiser to screen, so we pass through
	// rather than calling Armor with an empty string — armorPost stays
	// Bypassed (the same marker the permissive/no-template tier uses).
	//
	// SCOPE NOTE (MVP, W8 CR 2026-06-01): Model Armor does NOT screen raw
	// image bytes — image-content safety (NSFW / disallowed imagery in the
	// generated PNG) is OUT OF SCOPE for this text guardrail. A dedicated
	// image-safety classifier is a future follow-up. If the response carries
	// text ALONGSIDE the image, that text IS still screened below (the image
	// being present does not exempt accompanying text).
	// Armor POST screens user-facing TEXT only. Skip when the turn carries no
	// completion text: an image-only response (W8) OR a tool-call-only turn
	// (ADR-177 — the model asked to call a tool; functionCall args are model-
	// emitted, not user input, and are out of scope for the text guardrail in
	// v1). A turn with text alongside tool_calls/image still screens the text.
	noTextToScreen := vendorResp.Completion == "" &&
		(len(vendorResp.ImageBytes) > 0 || vendorResp.ToolCallsJSON != "")
	armorPost := ArmorVerdictBypassed
	// Declared here rather than at step 8b so the POST-block return below can
	// report it honestly: when the completion is refused we never reach the
	// tool-call leg, and Bypassed is exactly what "did not run" means.
	armorPostToolCalls := ArmorVerdictBypassed
	sanitisedCompletion := vendorResp.Completion
	if policy.ArmorTemplate != "" && !noTextToScreen {
		armorPost, sanitisedCompletion, err = e.armor.SanitizeModelResponse(ctx, policy.ArmorTemplate, vendorResp.Completion)
		if err != nil {
			return nil, &InvokeError{
				Reason: FinishReasonVendorError,
				Detail: "model armor POST call failed",
				Inner:  err,
			}
		}
		if armorPost.IsTerminalBlock() {
			// Emit full-cost ledger event (vendor billed for the output even
			// though we redacted it before returning). See note on the
			// armorPre terminal-block emitOutbox above for the rationale on
			// the explicit `_ =`.
			_ = e.emitOutbox(ctx, invocationID, req, policy, vendorResp.Usage, armorPre, armorPost, fallbackChain, vendorResp.ModelVersion)
			e.emitPolicyViolation(ctx, invocationID, req, policy, ArmorLegPost, armorPost)
			return &ExecuteResponse{
				InvocationID:       invocationID,
				ArmorPre:           armorPre,
				ArmorPreContents:   armorPreContents,
				ArmorPreTools:      armorPreTools,
				ArmorPost:          armorPost,
				ArmorPostToolCalls: armorPostToolCalls,
				Vendor:             string(policy.Vendor),
				ModelVersion:       vendorResp.ModelVersion,
				Usage:              vendorResp.Usage,
				FallbackChain:      fallbackChain,
				LatencyMs:          int32(e.now().Sub(start).Milliseconds()), // #nosec G115 -- elapsed ms since request start; bounded far below int32 (overflows only past ~24.8 days)
				FinishReason:       FinishReasonModelArmorBlock,
				FinishDetail:       "model armor POST blocked the response",
				CompletedAt:        e.now(),
				GatewayVersion:     e.gatewayVersion,
				AccountingState:    acctState,
			}, nil
		}
	}

	// Step 8b. Armor POST on the text DERIVED from tool_calls_json (G1'-2,
	// the tool-call half).
	//
	// Step 8 screens the completion and skips the turn entirely when there is
	// none. For an image-only response that is right; for a TOOL-CALL-only
	// response it is a hole, because the model's text moved into the arguments.
	// A prompt-injected model that cannot say a payload in its completion can
	// still put it in a tool argument, and nothing looked at it.
	//
	// It is a SEPARATE leg from step 8 rather than folded into it so governance
	// evidence can tell "the model's answer was refused" from "the model's tool
	// arguments were refused": different subjects, different remediation.
	//
	// POSTURE, owner's closing rule 2026-08-07, and the precedent step 6b set:
	// this screens content that has never been screened, so it is AUDIT-ONLY by
	// default. Tool-calling turns succeed today and none of them starts failing.
	// Promotion is EnforceToolCallScreen, a config flip, once there is measured
	// data.
	//
	// A permissive tier (empty template) adds no leg here either.
	//
	// The IMAGE half of this gap is deliberately NOT here: screening generated
	// image bytes needs a dedicated classifier and its own posture decision, and
	// an image-only turn passes through exactly as it did before.
	if policy.ArmorTemplate != "" {
		if derived := DeriveScreenableToolCallText(vendorResp.ToolCallsJSON); derived != "" {
			verdict, _, tErr := e.armor.SanitizeModelResponse(ctx, policy.ArmorTemplate, derived)
			switch {
			case tErr != nil:
				// A fault is not a verdict, and must never be recorded as
				// Allow. While the leg is audit-only a screening outage
				// degrades to unscreened rather than refusing a tool call the
				// agent is waiting on; once enforcing, it fails loud like any
				// other guardrail fault.
				if e.enforceToolCallScreen {
					return nil, &InvokeError{
						Reason: FinishReasonVendorError,
						Detail: "model armor POST (tool calls) call failed",
						Inner:  tErr,
					}
				}
				armorPostToolCalls = ArmorVerdictUnspecified
				slog.ErrorContext(ctx, "model gateway: tool-call screen faulted, turn proceeded UNSCREENED (audit-only leg)",
					"agent_id", req.AgentID, "tenant_id", req.TenantID,
					"invocation_id", invocationID, "error", tErr)
			default:
				armorPostToolCalls = verdict
				if verdict.IsTerminalBlock() {
					// Evidence is emitted in BOTH postures. The whole purpose
					// of the audit-only phase is to produce the measurement
					// that justifies promoting it.
					e.emitPolicyViolation(ctx, invocationID, req, policy, ArmorLegPostToolCalls, verdict)
					slog.WarnContext(ctx, "model gateway: tool-call screen BLOCKED",
						"agent_id", req.AgentID, "tenant_id", req.TenantID,
						"invocation_id", invocationID, "verdict", verdict.String(),
						"enforced", e.enforceToolCallScreen,
						"posture", map[bool]string{true: "refused", false: "audit-only, turn proceeded"}[e.enforceToolCallScreen])
					if e.enforceToolCallScreen {
						// Full-cost ledger event: the vendor billed for the
						// output even though we withhold it, the same
						// rationale as the step 8 terminal block above.
						_ = e.emitOutbox(ctx, invocationID, req, policy, vendorResp.Usage, armorPre, armorPost, fallbackChain, vendorResp.ModelVersion)
						return &ExecuteResponse{
							InvocationID:       invocationID,
							ArmorPre:           armorPre,
							ArmorPreContents:   armorPreContents,
							ArmorPreTools:      armorPreTools,
							ArmorPost:          armorPost,
							ArmorPostToolCalls: armorPostToolCalls,
							Vendor:             string(policy.Vendor),
							ModelVersion:       vendorResp.ModelVersion,
							Usage:              vendorResp.Usage,
							FallbackChain:      fallbackChain,
							LatencyMs:          int32(e.now().Sub(start).Milliseconds()), // #nosec G115 -- elapsed ms since request start; bounded far below int32
							FinishReason:       FinishReasonModelArmorBlock,
							FinishDetail:       "model armor POST blocked the tool calls",
							CompletedAt:        e.now(),
							GatewayVersion:     e.gatewayVersion,
							AccountingState:    acctState,
						}, nil
					}
				}
			}
		}
	}

	// Step 9 — debit budget + emit outbox event (transactional pattern).
	//
	// The debit + outbox insert run in ONE database transaction when the
	// atomic port is wired (transactional outbox pattern). A deduped
	// dispatch still enqueues the real cost; only the debit is suppressed.
	// Budget alert mode debits too (the alert is an observability signal,
	// not a debit exemption).
	//
	// → ACCOUNTED: the settlement commits here, in its OWN short transaction.
	// The provider dispatch (step 7) ran with NO DB transaction in flight.
	debit := vendorResp.Usage.CostMicros
	if deduped {
		debit = 0
	}
	settled := false
	if budget != nil {
		dbCtx, dbCancel := e.dbContext(ctx)
		if e.atomicOutbox != nil {
			evt := e.buildUsageEvent(invocationID, req, policy, vendorResp.Usage, armorPre, armorPost, fallbackChain, vendorResp.ModelVersion)
			if err := e.atomicOutbox.DebitAndEnqueue(dbCtx, evt, debit); err != nil {
				dbCancel()
				return nil, &InvokeError{
					Reason: FinishReasonVendorError,
					Detail: "budget debit + outbox enqueue failed",
					Inner:  err,
				}
			}
			settled = true
		} else {
			// Fallback: separate calls (non-atomic). Used when the atomic
			// port is not wired (unit tests, non-pg adapters).
			if debit > 0 {
				if err := e.budget.DebitSpent(dbCtx, req.TenantID, debit); err != nil {
					dbCancel()
					return nil, &InvokeError{
						Reason: FinishReasonVendorError,
						Detail: "budget debit failed",
						Inner:  err,
					}
				}
			}
			if err := e.emitOutbox(dbCtx, invocationID, req, policy, vendorResp.Usage, armorPre, armorPost, fallbackChain, vendorResp.ModelVersion); err != nil {
				dbCancel()
				return nil, &InvokeError{
					Reason: FinishReasonVendorError,
					Detail: "outbox enqueue failed",
					Inner:  err,
				}
			}
			settled = true
		}
		dbCancel()
	} else if e.atomicOutbox == nil {
		// No budget configured: still emit the outbox event (cost visibility).
		dbCtx, dbCancel := e.dbContext(ctx)
		if err := e.emitOutbox(dbCtx, invocationID, req, policy, vendorResp.Usage, armorPre, armorPost, fallbackChain, vendorResp.ModelVersion); err != nil {
			dbCancel()
			return nil, &InvokeError{
				Reason: FinishReasonVendorError,
				Detail: "outbox enqueue failed",
				Inner:  err,
			}
		}
		dbCancel()
		settled = true
	}
	if settled {
		acctState = AccountingAccounted
	}

	// Step 10 — mana debit (OPTIONAL). When the mana port is wired the
	// Executor debits the per-GCID wallet post-success, idempotent on the
	// invocation id. When nil the transport-level middleware
	// (adapter/middleware.ManaMetering) is the gate and the Executor must not
	// double-gate. A debit error is logged loudly but NEVER un-serves a
	// completed call (resilience over strictness — the same contract the
	// middleware holds).
	if e.mana != nil && !deduped {
		d, derr := e.mana.Debit(ctx, req.GCID, req.TenantID, actionCode, invocationID)
		switch {
		case derr != nil:
			slog.ErrorContext(ctx, "model gateway: post-success mana debit error — served-but-uncharged",
				"action_code", actionCode, "gcid", req.GCID, "invocation_id", invocationID, "err", derr)
		case d.UnknownAction:
			slog.InfoContext(ctx, "model gateway: unpriced action_code at debit → un-metered",
				"action_code", actionCode, "gcid", req.GCID)
		case !d.Success:
			slog.WarnContext(ctx, "model gateway: post-success mana debit shortfall — served + audited",
				"action_code", actionCode, "gcid", req.GCID,
				"required", d.RequiredUnits, "balance_after", d.BalanceAfterUnits)
		}
	}

	// Step 11 — happy-path return. Image fields are carried verbatim from the
	// vendor response (empty for text-only responses) per W8, CR 2026-06-01.
	return &ExecuteResponse{
		InvocationID:       invocationID,
		Completion:         sanitisedCompletion,
		ImageBytes:         vendorResp.ImageBytes,
		ImageMIMEType:      vendorResp.ImageMIMEType,
		RevisedPrompt:      vendorResp.RevisedPrompt,
		ToolCallsJSON:      vendorResp.ToolCallsJSON,
		Citations:          vendorResp.Citations,
		SearchQueries:      vendorResp.SearchQueries,
		Usage:              vendorResp.Usage,
		Vendor:             string(policy.Vendor),
		ModelVersion:       vendorResp.ModelVersion,
		ArmorPre:           armorPre,
		ArmorPreContents:   armorPreContents,
		ArmorPreTools:      armorPreTools,
		ArmorPost:          armorPost,
		ArmorPostToolCalls: armorPostToolCalls,
		FallbackChain:      fallbackChain,
		LatencyMs:          int32(e.now().Sub(start).Milliseconds()), // #nosec G115 -- elapsed ms since request start; bounded far below int32 (overflows only past ~24.8 days)
		FinishReason:       vendorResp.FinishReason,
		FinishDetail:       vendorResp.FinishDetail,
		CompletedAt:        e.now(),
		GatewayVersion:     e.gatewayVersion,
		Deduped:            deduped,
		AccountingState:    acctState,
	}, nil
}

// fallbackInvokeError converts the FallbackEngine's terminal error into the
// transport-facing InvokeError.
//
// A registry/capability/credential refusal is a gateway CONFIGURATION fault, so
// its own detail is surfaced — the operator needs to know which model or
// credential is wrong, not that "all targets failed". Every other failure gets
// the walk summary, carrying the LAST target's upstream status: a fallback
// that fails without an upstream status yields no relay even when an earlier
// target carried one (mirrors the Python walk).
func (e *Executor) fallbackInvokeError(err error) error {
	var exhaustedErr *FallbackExhaustedError
	if !errors.As(err, &exhaustedErr) {
		return &InvokeError{
			Reason: FinishReasonVendorError,
			Detail: "provider dispatch failed",
			Inner:  err,
		}
	}
	last := exhaustedErr.Last
	if detail := preflightDetail(last.Err); detail != "" {
		return &InvokeError{
			Reason:         FinishReasonVendorError,
			Detail:         detail,
			Inner:          last.Err,
			UpstreamStatus: last.UpstreamStatus,
			Kind:           ErrorKindConfig,
		}
	}
	detail := fmt.Sprintf("all %d provider attempts failed; last provider %q", len(exhaustedErr.Attempts), last.Target.Vendor)
	if last.UpstreamStatus != nil {
		detail = fmt.Sprintf("%s; upstream status %d", detail, *last.UpstreamStatus)
	}
	return &InvokeError{
		Reason:         FinishReasonVendorError,
		Detail:         detail,
		Inner:          last.Err,
		UpstreamStatus: last.UpstreamStatus,
		Kind:           exhaustedErr.Kind,
	}
}

// preflightDetail returns the raw detail of a registry/capability/credential
// refusal (the three preflight faults an operator must be able to read
// directly), or "" when the error is not one of them. An unregistered vendor
// family is deliberately NOT in this set: it is reported through the walk
// summary, which names the family.
func preflightDetail(err error) string {
	if err == nil {
		return ""
	}
	var cfgErr *ConfigError
	var capErr *CapabilityError
	var credErr *CredentialError
	if errors.As(err, &cfgErr) || errors.As(err, &capErr) || errors.As(err, &credErr) {
		return err.Error()
	}
	return ""
}

// dbContext returns a context bounded by the DB timeout. Every database
// operation (budget read, idempotency claim, settlement debit + outbox) runs
// under this bound so a slow DB cannot consume the entire request budget —
// the provider dispatch is the expensive part and needs the remaining time.
// A zero dbTimeout means no DB timeout (the request context is the only
// bound).
func (e *Executor) dbContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if e.dbTimeout <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, e.dbTimeout)
}

// emitOutbox builds the TokenUsageEvent + hands it to the outbox port.
// Factored out so the Armor PRE/POST short-circuit paths can reuse it
// for partial-cost emission.
func (e *Executor) emitOutbox(
	ctx context.Context,
	invocationID string,
	req ExecuteRequest,
	policy AgentPolicy,
	usage TokenUsage,
	armorPre ArmorVerdict,
	armorPost ArmorVerdict,
	fallbackChain []string,
	modelVersion string,
) error {
	evt := e.buildUsageEvent(invocationID, req, policy, usage, armorPre, armorPost, fallbackChain, modelVersion)
	return e.outbox.EnqueueTokenUsageRecorded(ctx, evt)
}

// buildUsageEvent builds the TokenUsageEvent for the outbox.
func (e *Executor) buildUsageEvent(
	invocationID string,
	req ExecuteRequest,
	policy AgentPolicy,
	usage TokenUsage,
	armorPre ArmorVerdict,
	armorPost ArmorVerdict,
	fallbackChain []string,
	modelVersion string,
) TokenUsageEvent {
	return TokenUsageEvent{
		UsageID:        invocationID,
		TenantID:       req.TenantID,
		GCID:           req.GCID,
		ModelID:        modelVersion,
		InputTokens:    usage.InputTokens,
		OutputTokens:   usage.OutputTokens,
		CachedTokens:   usage.CachedTokens,
		CostMicros:     usage.CostMicros,
		InvocationID:   invocationID,
		AgentRole:      req.AgentID,
		AgentID:        req.AgentID,
		ActionCode:     actionCodeOrAgent(req.ActionCode, req.AgentID),
		RecordedAt:     e.now(),
		Vendor:         string(policy.Vendor),
		FallbackChain:  fallbackChain,
		ArmorPre:       armorPre,
		ArmorPost:      armorPost,
		GatewayVersion: e.gatewayVersion,
		Traceparent:    req.Traceparent,
		Tracestate:     req.Tracestate,
	}
}

// actionCodeOrAgent mirrors the Python reference's `request.get("action_code")
// or agent`: an explicit per-turn action_code wins, else the agent id labels
// the usage.
func actionCodeOrAgent(actionCode, agentID string) string {
	if actionCode != "" {
		return actionCode
	}
	return agentID
}

// clampMaxTokens clamps the max_tokens in the generation config to the
// model's max_output_tokens ceiling. Returns nil if no clamping is needed.
func clampMaxTokens(cfg map[string]any, ceiling int) map[string]any {
	if ceiling <= 0 || len(cfg) == 0 {
		return nil
	}
	raw, ok := cfg["max_tokens"]
	if !ok {
		return nil
	}
	var requested float64
	switch v := raw.(type) {
	case float64:
		requested = v
	case int:
		requested = float64(v)
	case int32:
		requested = float64(v)
	case int64:
		requested = float64(v)
	default:
		return nil
	}
	if requested <= float64(ceiling) {
		return nil
	}
	out := make(map[string]any, len(cfg))
	for k, v := range cfg {
		out[k] = v
	}
	out["max_tokens"] = ceiling
	return out
}

// emitPolicyViolation publishes exactly one PolicyViolationDetected event for a
// call Cloud Model Armor refused (ADR-152 amendment 2026-08-07, requirement G2).
//
// Called from the PRE and POST short-circuits, which are mutually exclusive by
// construction: a PRE block returns before dispatch, so POST is only ever
// reached when PRE allowed. One blocked request therefore yields one event, not
// one per leg evaluated.
//
// ONLY ArmorVerdictBlock emits. IsTerminalBlock() is also true for
// ArmorVerdictError, but an ERROR verdict means Armor could not classify the
// text, not that a filter matched. Recording it as a detected violation would
// fabricate governance evidence. The ERROR refusal stays observable through the
// token-usage ledger, which carries the verdict enum verbatim.
//
// FAIL-LOUD BUT NON-BLOCKING. The caller's refusal is the user-visible outcome
// and must not be delayed or broken by a governance-lane fault, so a publisher
// error is logged at ERROR with enough context to reconcile the missing row
// (tenant, gcid, agent, invocation, leg) and then swallowed. A nil publisher is
// logged the same way rather than passing silently: an unwired deploy screams
// on its first block instead of losing evidence without a trace.
func (e *Executor) emitPolicyViolation(
	ctx context.Context,
	invocationID string,
	req ExecuteRequest,
	policy AgentPolicy,
	leg ArmorLeg,
	verdict ArmorVerdict,
) {
	if verdict != ArmorVerdictBlock {
		return
	}
	evt := PolicyViolationEvent{
		ViolationID:   e.newID(),
		TenantID:      req.TenantID,
		GCID:          req.GCID,
		AgentID:       req.AgentID,
		InvocationID:  invocationID,
		Leg:           leg,
		ArmorTemplate: policy.ArmorTemplate,
		Verdict:       verdict,
		DetectedAt:    e.now(),
		Traceparent:   req.Traceparent,
		Tracestate:    req.Tracestate,
	}
	if e.violations == nil {
		slog.ErrorContext(ctx, "model gateway: Armor BLOCK not published, no ViolationPublisher wired",
			"tenant_id", req.TenantID,
			"gcid", req.GCID,
			"agent_id", req.AgentID,
			"invocation_id", invocationID,
			"armor_leg", string(leg),
			"violation_id", evt.ViolationID,
		)
		return
	}
	if err := e.violations.EnqueuePolicyViolationDetected(ctx, evt); err != nil {
		slog.ErrorContext(ctx, "model gateway: PolicyViolationDetected publish failed",
			"err", err,
			"tenant_id", req.TenantID,
			"gcid", req.GCID,
			"agent_id", req.AgentID,
			"invocation_id", invocationID,
			"armor_leg", string(leg),
			"violation_id", evt.ViolationID,
		)
	}
}
