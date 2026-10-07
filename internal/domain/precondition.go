package domain

import "fmt"

// PreconditionError is a typed Invoke refusal raised BEFORE any vendor call and
// before any mana debit (ADR-254 D7 / ADR-252 D5). The gRPC adapter maps it to
// FAILED_PRECONDITION (or UNAVAILABLE for DenySuspensionUnreadable) and the
// status message leads with the machine token, so a caller can prefix-match
// the reason off the status without a structured field on the wire.
type PreconditionError struct {
	// Reason is one of the Deny* tokens below.
	Reason string
	Detail string
	Inner  error
}

func (e *PreconditionError) Error() string {
	if e.Inner != nil {
		return fmt.Sprintf("%s: %s: %v", e.Reason, e.Detail, e.Inner)
	}
	return fmt.Sprintf("%s: %s", e.Reason, e.Detail)
}

func (e *PreconditionError) Unwrap() error { return e.Inner }

// Deny* machine tokens (ADR-254 D7 names them; ADR-252 D5 makes
// companion_suspended the sibling of kill_switch_engaged).
const (
	// DenySurfaceUnstamped: the caller did not stamp InvokeRequest.surface.
	// An absent surface is a LOUD condition (ADR-252 Q1), never "not a
	// companion turn": an unstamped path must be visible, not quietly exempt.
	DenySurfaceUnstamped = "surface_unstamped"
	// DenyCompanionSuspended: an operator has contained the companion for this
	// call's scope (platform, tenant, or skill). Refused honestly, costs nothing.
	DenyCompanionSuspended = "companion_suspended"
	// DenySuspensionUnreadable: the containment state could not be read. A
	// companion whose containment state cannot be read must not run (ADR-252
	// D6: unreadable denies; empty allows). Mapped to UNAVAILABLE, not
	// FAILED_PRECONDITION, because it is a store fault rather than a verdict.
	DenySuspensionUnreadable = "suspension_unreadable"
)

// CompanionSurfaces is the ADR-254 D7 set for "all companion turns": the
// platform-scope suspension with skill_key NULL governs exactly these surfaces.
var CompanionSurfaces = map[string]struct{}{
	"companion_chat":      {},
	"companion_diagnosis": {},
}

// IsCompanionSurface reports whether a stamped surface is a Learning Companion
// turn surface governed by ADR-252 containment.
func IsCompanionSurface(surface string) bool {
	_, ok := CompanionSurfaces[surface]
	return ok
}
