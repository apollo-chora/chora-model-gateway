package clients

import (
	"context"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// DomainManaMeter adapts *ManaClient to domain.ManaMeter — the domain-typed
// mana port the GroundedSearch chain uses (ADR-231). The Invoke path uses the
// middleware.ManaPort view of the SAME underlying client; both speak the one
// identity ManaService (ADR-177 central metering). This wrapper only re-shapes
// the result structs (middleware.* → domain.*) so the domain stays free of the
// middleware/adapter import.
type DomainManaMeter struct{ c *ManaClient }

// NewDomainManaMeter wraps a ManaClient as a domain.ManaMeter. c MUST be non-nil.
func NewDomainManaMeter(c *ManaClient) *DomainManaMeter { return &DomainManaMeter{c: c} }

var _ domain.ManaMeter = (*DomainManaMeter)(nil)

// Quote runs the dry-run affordability check and returns the domain view.
func (m *DomainManaMeter) Quote(ctx context.Context, gcid, tenantID, actionCode string) (domain.ManaQuote, error) {
	q, err := m.c.Quote(ctx, gcid, tenantID, actionCode)
	if err != nil {
		return domain.ManaQuote{}, err
	}
	return domain.ManaQuote{
		Affordable:     q.Affordable,
		RequiredUnits:  q.RequiredUnits,
		AvailableUnits: q.AvailableUnits,
		UnknownAction:  q.UnknownAction,
	}, nil
}

// Debit charges the catalogue cost and returns the domain view.
func (m *DomainManaMeter) Debit(ctx context.Context, gcid, tenantID, actionCode, idemKey string) (domain.ManaDebit, error) {
	d, err := m.c.Debit(ctx, gcid, tenantID, actionCode, idemKey)
	if err != nil {
		return domain.ManaDebit{}, err
	}
	return domain.ManaDebit{
		Success:           d.Success,
		RequiredUnits:     d.RequiredUnits,
		BalanceAfterUnits: d.BalanceAfterUnits,
		UnknownAction:     d.UnknownAction,
	}, nil
}
