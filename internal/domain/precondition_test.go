package domain_test

import (
	"errors"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
)

// ADR-254 D7 / ADR-252 D5: a denied Invoke carries a machine token FIRST in its
// message so callers (the agents' modelgatewayclient) can prefix-match it off
// the gRPC status without a structured field on the wire.
func TestPreconditionError_TokenFirst(t *testing.T) {
	err := &domain.PreconditionError{Reason: domain.DenyCompanionSuspended, Detail: "tenant scope, skill companion_chat_turn_basic"}
	assert.Equal(t, "companion_suspended: tenant scope, skill companion_chat_turn_basic", err.Error())

	inner := errors.New("pq: relation does not exist")
	werr := &domain.PreconditionError{Reason: domain.DenySuspensionUnreadable, Detail: "read platform_companion_suspension", Inner: inner}
	assert.Equal(t, "suspension_unreadable: read platform_companion_suspension: pq: relation does not exist", werr.Error())
	assert.True(t, errors.Is(werr, inner), "Unwrap must expose the inner cause")
}

func TestPreconditionError_Tokens(t *testing.T) {
	// The tokens are a wire contract (ADR-254 D7 names them); pin them.
	assert.Equal(t, "surface_unstamped", domain.DenySurfaceUnstamped)
	assert.Equal(t, "companion_suspended", domain.DenyCompanionSuspended)
	assert.Equal(t, "suspension_unreadable", domain.DenySuspensionUnreadable)
}

func TestIsCompanionSurface(t *testing.T) {
	// ADR-254 D7: "all companion turns" = surfaces {companion_chat, companion_diagnosis}.
	assert.True(t, domain.IsCompanionSurface("companion_chat"))
	assert.True(t, domain.IsCompanionSurface("companion_diagnosis"))
	assert.False(t, domain.IsCompanionSurface("kg_exploration"))
	assert.False(t, domain.IsCompanionSurface("qgen"))
	assert.False(t, domain.IsCompanionSurface(""))
}
