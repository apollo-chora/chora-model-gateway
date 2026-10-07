package clients_test

import (
	"context"
	"testing"

	identityv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/identity/v1"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/clients"
)

// DomainManaMeter is a thin re-shape of ManaClient (middleware.* → domain.*)
// for the GroundedSearch chain. These tests confirm the passthrough.

func TestDomainManaMeter_Quote_Affordable(t *testing.T) {
	g := &fakeGRPC{resp: &identityv1.DeductManaResponse{Success: true, BalanceAfterUnits: 420}}
	m := clients.NewDomainManaMeter(clients.NewManaClient(g))

	q, err := m.Quote(context.Background(), "gcid", "tenant", "familiar_far_sight_grounded_search")
	if err != nil {
		t.Fatalf("Quote err: %v", err)
	}
	if !q.Affordable {
		t.Errorf("Affordable = false, want true")
	}
	if q.AvailableUnits != 420 {
		t.Errorf("AvailableUnits = %d, want 420", q.AvailableUnits)
	}
}

func TestDomainManaMeter_Quote_Shortfall(t *testing.T) {
	g := &fakeGRPC{resp: &identityv1.DeductManaResponse{Success: false, RequiredUnits: 80, CurrentBalanceUnits: 10}}
	m := clients.NewDomainManaMeter(clients.NewManaClient(g))

	q, err := m.Quote(context.Background(), "gcid", "tenant", "familiar_far_sight_grounded_search")
	if err != nil {
		t.Fatalf("Quote err: %v", err)
	}
	if q.Affordable {
		t.Errorf("Affordable = true, want false")
	}
	if q.RequiredUnits != 80 || q.AvailableUnits != 10 {
		t.Errorf("required/available = %d/%d, want 80/10", q.RequiredUnits, q.AvailableUnits)
	}
}

func TestDomainManaMeter_Debit_Success(t *testing.T) {
	g := &fakeGRPC{resp: &identityv1.DeductManaResponse{Success: true, BalanceAfterUnits: 340}}
	m := clients.NewDomainManaMeter(clients.NewManaClient(g))

	d, err := m.Debit(context.Background(), "gcid", "tenant", "familiar_far_sight_grounded_search", "idem-1")
	if err != nil {
		t.Fatalf("Debit err: %v", err)
	}
	if !d.Success {
		t.Errorf("Success = false, want true")
	}
	if d.BalanceAfterUnits != 340 {
		t.Errorf("BalanceAfterUnits = %d, want 340", d.BalanceAfterUnits)
	}
}
