package domain_test

import (
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ----------------------------------------------------------------------------
// BudgetRequiredTenants — the tenant-scoped fail-closed override
// (CHORA_LLM_BUDGET_REQUIRED_TENANTS).
// ----------------------------------------------------------------------------

func TestParseBudgetRequiredTenants_ValidListCanonicalised(t *testing.T) {
	set, err := domain.ParseBudgetRequiredTenants("33333333-3333-7333-8333-333333333333, 44444444-4444-7444-8444-444444444444 ")
	require.NoError(t, err)
	assert.Len(t, set, 2)
	assert.True(t, set.Contains("33333333-3333-7333-8333-333333333333"))
	assert.True(t, set.Contains("44444444-4444-7444-8444-444444444444"))
}

// The lookup is canonical: an uppercase probe matches a lowercase stored key.
func TestBudgetRequiredTenants_ContainsCanonicalisesProbe(t *testing.T) {
	set, err := domain.ParseBudgetRequiredTenants("33333333-3333-7333-8333-333333333333")
	require.NoError(t, err)
	assert.True(t, set.Contains("33333333-3333-7333-8333-333333333333"))
	assert.False(t, set.Contains("44444444-4444-7444-8444-444444444444"))
}

// A non-UUID tenant id is never contained — the override can only narrow the
// fail-open default, never widen it.
func TestBudgetRequiredTenants_NonUuidTenantNeverContained(t *testing.T) {
	set, err := domain.ParseBudgetRequiredTenants("33333333-3333-7333-8333-333333333333")
	require.NoError(t, err)
	assert.False(t, set.Contains("tenant-1"))
	assert.False(t, set.Contains(""))
}

// Unset / blank ⇒ empty set, no error.
func TestParseBudgetRequiredTenants_Empty(t *testing.T) {
	for _, raw := range []string{"", "   "} {
		set, err := domain.ParseBudgetRequiredTenants(raw)
		require.NoError(t, err)
		assert.Len(t, set, 0)
	}
}

// A malformed entry is an error, never a silent skip — a typo must not
// silently un-gate a tenant.
func TestParseBudgetRequiredTenants_Malformed(t *testing.T) {
	_, err := domain.ParseBudgetRequiredTenants("33333333-3333-7333-8333-333333333333,not-a-uuid")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "CHORA_LLM_BUDGET_REQUIRED_TENANTS")
	assert.Contains(t, err.Error(), "not-a-uuid")
}

// An empty slot (trailing comma, blank between commas) is an error too.
func TestParseBudgetRequiredTenants_EmptyEntry(t *testing.T) {
	for _, raw := range []string{
		"33333333-3333-7333-8333-333333333333,",
		"33333333-3333-7333-8333-333333333333,,44444444-4444-7444-8444-444444444444",
		"33333333-3333-7333-8333-333333333333, ,",
	} {
		_, err := domain.ParseBudgetRequiredTenants(raw)
		require.Error(t, err, "raw %q must fail", raw)
		assert.Contains(t, err.Error(), "CHORA_LLM_BUDGET_REQUIRED_TENANTS")
	}
}
