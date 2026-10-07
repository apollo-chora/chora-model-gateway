package domain

// ArmorVerdict mirrors the chora.services.model_gateway.v1.ModelArmorVerdict
// enum from chora-contracts. Kept in domain (vs ports) because it is a
// pure value type with no infra dependency — the proto enum value flows
// through unchanged from the wire to the ledger event.
type ArmorVerdict int

const (
	ArmorVerdictUnspecified ArmorVerdict = 0
	ArmorVerdictAllow       ArmorVerdict = 1
	ArmorVerdictBlock       ArmorVerdict = 2
	ArmorVerdictSanitise    ArmorVerdict = 3
	ArmorVerdictError       ArmorVerdict = 4
	ArmorVerdictBypassed    ArmorVerdict = 5
)

// IsTerminalBlock returns true when the verdict means "refuse the call".
// Used by the service layer to short-circuit before vendor dispatch (on
// PRE) or before outbox emission of a completed response (on POST).
func (v ArmorVerdict) IsTerminalBlock() bool {
	return v == ArmorVerdictBlock || v == ArmorVerdictError
}

// String returns a human-readable name for OTLP span attributes + logs.
func (v ArmorVerdict) String() string {
	switch v {
	case ArmorVerdictAllow:
		return "allow"
	case ArmorVerdictBlock:
		return "block"
	case ArmorVerdictSanitise:
		return "sanitise"
	case ArmorVerdictError:
		return "error"
	case ArmorVerdictBypassed:
		return "bypassed"
	default:
		return "unspecified"
	}
}
