package domain

import (
	"fmt"
	"strings"

	"github.com/google/uuid"
)

// BudgetRequiredTenants is the tenant-scoped override for the fail-closed
// "budget required" mode (CHORA_LLM_BUDGET_REQUIRED_TENANTS): a
// comma-separated list of tenant UUIDs for which a missing active budget
// window BLOCKS provider calls even when the global CHORA_LLM_BUDGET_REQUIRED
// flag is OFF. Tenants not in the set keep the historical fail-open behaviour
// until their budgets have been provisioned, so a shared deployment can gate
// one designated (demo) tenant without disturbing every other tenant that
// legitimately has no budget rows.
type BudgetRequiredTenants map[string]struct{}

// Contains reports whether tenantID is in the set. The lookup is canonical:
// the probe is normalised through uuid.Parse just like the stored keys, so
// case and formatting differences do not split a tenant across two entries.
// A tenantID that is not a parseable UUID is never contained — the override
// can only ever narrow the fail-open default, never widen it.
func (s BudgetRequiredTenants) Contains(tenantID string) bool {
	canonical, err := uuid.Parse(strings.TrimSpace(tenantID))
	if err != nil {
		return false
	}
	_, ok := s[canonical.String()]
	return ok
}

// ParseBudgetRequiredTenants parses CHORA_LLM_BUDGET_REQUIRED_TENANTS. Strict,
// mirroring the CHORA_DEMO_MANA_ALLOWED_GCIDS contract in chora-identity: every
// comma-separated entry must be a non-empty, parseable UUID. A malformed
// entry or an empty slot (trailing comma, blank between commas) is an error,
// never a silent skip — a typo must not silently un-gate a tenant.
func ParseBudgetRequiredTenants(raw string) (BudgetRequiredTenants, error) {
	set := make(BudgetRequiredTenants)
	if strings.TrimSpace(raw) == "" {
		return set, nil
	}
	for _, part := range strings.Split(raw, ",") {
		entry := strings.TrimSpace(part)
		if entry == "" {
			return nil, fmt.Errorf("CHORA_LLM_BUDGET_REQUIRED_TENANTS has an empty entry — every entry must be a non-empty tenant UUID")
		}
		canonical, err := uuid.Parse(entry)
		if err != nil {
			return nil, fmt.Errorf("CHORA_LLM_BUDGET_REQUIRED_TENANTS has malformed entry %q: %w", entry, err)
		}
		set[canonical.String()] = struct{}{}
	}
	return set, nil
}

// budgetRequiredFor reports whether the fail-closed "budget required" mode
// applies to this tenant: the global flag (CHORA_LLM_BUDGET_REQUIRED) fails
// every tenant closed; otherwise only the explicitly listed tenants
// (CHORA_LLM_BUDGET_REQUIRED_TENANTS) are fail-closed.
func budgetRequiredFor(global bool, tenants BudgetRequiredTenants, tenantID string) bool {
	return global || tenants.Contains(tenantID)
}
