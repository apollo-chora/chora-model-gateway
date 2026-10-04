package domain

import "testing"

// TestBudgetState_Decide is the flow-control table. Two axes matter and they
// are not interchangeable: whether the budget is exhausted, and what the
// tenant's policy says to do about it. The policy only has any say once the
// ceiling is actually hit — below it, every policy proceeds untouched.
func TestBudgetState_Decide(t *testing.T) {
	cases := []struct {
		name    string
		budget  int64
		spent   int64
		policy  BudgetPolicy
		want    BudgetDecision
		explain string
	}{
		// Below the ceiling: the policy is irrelevant, the call proceeds on the
		// model the caller asked for.
		{"under block", 1000, 0, BudgetPolicyBlock, BudgetAllow, "a block policy only bites at the ceiling"},
		{"under downgrade", 1000, 999, BudgetPolicyDowngrade, BudgetAllow, "downgrading early would cost the tenant quality it has paid for"},
		{"under alert", 1000, 1, BudgetPolicyAlert, BudgetAllow, "nothing to alert about yet"},
		{"under unknown policy", 1000, 1, BudgetPolicy("something-new"), BudgetAllow, "an unknown policy must not invent a refusal that the tenant never asked for"},

		// At or past the ceiling.
		{"at ceiling block", 1000, 1000, BudgetPolicyBlock, BudgetBlock, "spent == budget is exhausted"},
		{"past ceiling block", 1000, 5000, BudgetPolicyBlock, BudgetBlock, "overspend is still a block"},
		{"at ceiling downgrade", 1000, 1000, BudgetPolicyDowngrade, BudgetDowngrade, "swap to the cheaper model rather than refusing"},
		{"past ceiling downgrade", 1000, 5000, BudgetPolicyDowngrade, BudgetDowngrade, "the ceiling does not become permeable past it"},
		{"at ceiling alert", 1000, 1000, BudgetPolicyAlert, BudgetAlert, "an exempt tenant still wants the O+ signal"},
		{"past ceiling alert", 1000, 5000, BudgetPolicyAlert, BudgetAlert, "same past the ceiling"},

		// Fail-safe: anything the code does not recognise is treated as the
		// strictest policy. A policy value that drifts (a new enum row in
		// chora_observability, a typo, an empty string from a nullable column)
		// must never quietly become "allow" — that is unbilled spend nobody
		// approved.
		{"empty policy", 1000, 1000, BudgetPolicy(""), BudgetBlock, "empty is not a licence to spend"},
		{"unknown policy", 1000, 1000, BudgetPolicy("pause"), BudgetBlock, "an unrecognised policy blocks"},
		{"misspelled block", 1000, 1000, BudgetPolicy("blok"), BudgetBlock, "a typo in the enum must not disable the ceiling"},
		{"case-mismatched block", 1000, 1000, BudgetPolicy("BLOCK"), BudgetBlock, "the enum is lowercase; a mismatch must not read as allow"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := BudgetState{BudgetUSDMicros: tc.budget, SpentUSDMicros: tc.spent, Policy: tc.policy}
			if got := b.Decide(); got != tc.want {
				t.Errorf("Decide = %v, want %v (%s)", got, tc.want, tc.explain)
			}
		})
	}
}

// TestBudgetState_DecideIgnoresTheDowngradeTarget pins the boundary between the
// decision and the action. Decide is pure flow control: choosing WHICH cheaper
// model to swap to is the service layer's job, and it re-resolves that model
// through the policy loader. A Decide that consulted the target would be
// resolving names it has no registry for.
func TestBudgetState_DecideIgnoresTheDowngradeTarget(t *testing.T) {
	withTarget := BudgetState{
		BudgetUSDMicros:  100,
		SpentUSDMicros:   100,
		Policy:           BudgetPolicyDowngrade,
		DowngradeToModel: "chora-cheap",
	}
	withoutTarget := withTarget
	withoutTarget.DowngradeToModel = ""

	if got, want := withTarget.Decide(), withoutTarget.Decide(); got != want {
		t.Errorf("Decide = %v with a downgrade target and %v without; the decision must not depend on it", got, want)
	}
	if withTarget.Decide() != BudgetDowngrade {
		t.Errorf("Decide = %v, want BudgetDowngrade", withTarget.Decide())
	}

	// A downgrade policy with no target is still a downgrade decision; the
	// missing model surfaces later, at the re-resolve, with the model's name in
	// the message — not here as a silently-allowed call.
	if withoutTarget.Decide() != BudgetDowngrade {
		t.Errorf("Decide with an empty downgrade target = %v, want BudgetDowngrade", withoutTarget.Decide())
	}
}

func TestBudgetState_IsExhausted(t *testing.T) {
	cases := []struct {
		name   string
		budget int64
		spent  int64
		want   bool
	}{
		{"empty budget", 0, 0, true},
		{"under", 1000, 999, false},
		{"exactly at", 1000, 1000, true},
		{"one micro over", 1000, 1001, true},
		{"far over", 1000, 1_000_000, true},
		{"zero ceiling with spend", 0, 1, true},
		// A negative ceiling is nonsense data, but treating it as "not
		// exhausted" would be the dangerous direction: the comparison is
		// unsigned-agnostic and fails toward the refusal.
		{"negative ceiling", -1, 0, true},
		{"negative spend", 1000, -5, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := BudgetState{BudgetUSDMicros: tc.budget, SpentUSDMicros: tc.spent}
			if got := b.IsExhausted(); got != tc.want {
				t.Errorf("IsExhausted = %v, want %v", got, tc.want)
			}
		})
	}
}

// TestBudgetState_ZeroCeilingIsExhausted documents the one case an operator is
// most likely to be surprised by. There is no "unlimited" sentinel in this
// table: an unlimited tenant has no active window, and the pg adapter returns
// (nil, nil) for that, which the service treats as allow. A row that exists
// with a zero ceiling is a zero budget.
func TestBudgetState_ZeroCeilingIsExhausted(t *testing.T) {
	b := BudgetState{BudgetUSDMicros: 0, SpentUSDMicros: 0, Policy: BudgetPolicyBlock}
	if !b.IsExhausted() {
		t.Error("a zero ceiling reported as un-exhausted")
	}
	if got := b.Decide(); got != BudgetBlock {
		t.Errorf("Decide = %v, want BudgetBlock for a zero ceiling under the block policy", got)
	}
	// Under downgrade it still downgrades rather than blocking: the policy, not
	// the zero, decides what an exhausted budget means.
	down := b
	down.Policy = BudgetPolicyDowngrade
	if got := down.Decide(); got != BudgetDowngrade {
		t.Errorf("Decide = %v, want BudgetDowngrade", got)
	}
}

// TestBudgetPolicy_Values pins the wire values. These strings are what
// chora_observability.llm_budget_policy stores, so a rename here would silently
// stop matching the database and make every tenant fall through to the
// fail-safe block.
func TestBudgetPolicy_Values(t *testing.T) {
	cases := map[BudgetPolicy]string{
		BudgetPolicyBlock:     "block",
		BudgetPolicyDowngrade: "downgrade",
		BudgetPolicyAlert:     "alert",
	}
	for policy, want := range cases {
		if string(policy) != want {
			t.Errorf("policy constant = %q, want %q (it is a database enum value)", policy, want)
		}
	}
}
