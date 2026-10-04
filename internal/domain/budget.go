package domain

// BudgetPolicy mirrors chora_observability.llm_budget_policy ENUM from
// migration 0008. Kept in domain because the gateway service decides
// flow based on it.
type BudgetPolicy string

const (
	BudgetPolicyBlock     BudgetPolicy = "block"
	BudgetPolicyDowngrade BudgetPolicy = "downgrade"
	BudgetPolicyAlert     BudgetPolicy = "alert"
)

// BudgetState is the per-tenant LLM budget snapshot the BudgetRepo port
// loads at the start of every Invoke. Populated from the
// chora_observability.per_tenant_llm_budget table.
type BudgetState struct {
	TenantID         string
	BudgetUSDMicros  int64
	SpentUSDMicros   int64
	Policy           BudgetPolicy
	DowngradeToModel LogicalModelID // empty for non-downgrade policies
}

// IsExhausted reports whether spent >= budget.
func (b BudgetState) IsExhausted() bool {
	return b.SpentUSDMicros >= b.BudgetUSDMicros
}

// BudgetDecision is the gateway's flow control derived from the
// BudgetState + policy.
type BudgetDecision int

const (
	// BudgetAllow — proceed with the originally-resolved model.
	BudgetAllow BudgetDecision = iota

	// BudgetDowngrade — proceed but switch to the policy.downgrade_to model.
	// The gateway's Invoke flow swaps the policy.ResolvedLogicalModelID
	// before vendor dispatch + records the downgrade in
	// InvokeResponse.fallback_chain.
	BudgetDowngrade

	// BudgetBlock — refuse the call. Gateway returns FAILED_PRECONDITION
	// + FinishReason BUDGET_BLOCK; no vendor dispatch; no outbox emit
	// (no successful call to record).
	BudgetBlock

	// BudgetAlert — proceed, but emit a high-priority OTLP violation
	// span + Pub/Sub event for the O+ dashboard. Used for "alert mode"
	// tenants who are contractually exempt from block but still want
	// observability.
	BudgetAlert
)

// Decide computes the BudgetDecision from a BudgetState.
//
// Pure function — testable with no infra dependency. The service layer
// calls this immediately after BudgetRepo.GetTenantBudget returns.
func (b BudgetState) Decide() BudgetDecision {
	if !b.IsExhausted() {
		return BudgetAllow
	}
	switch b.Policy {
	case BudgetPolicyDowngrade:
		return BudgetDowngrade
	case BudgetPolicyAlert:
		return BudgetAlert
	default:
		// "block" is also the default for unrecognised policies — fail safe
		// (refuse rather than allow).
		return BudgetBlock
	}
}
