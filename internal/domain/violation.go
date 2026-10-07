// violation.go: the ADR-152 governance-violation producer (amendment
// 2026-08-07, requirement G2).
//
// A Model Armor BLOCK at the gateway is the only moment where the verdict, the
// tenant context, the learner GCID and the agent identity exist together. The
// gateway is the single un-bypassable LLM chokepoint (ADR-163), so it is the
// producer of record for the event.
//
// The originally ratified ADR-152 shape (a Cloud Logging sink from the Model
// Armor audit log straight onto the topic) is UNBUILDABLE: the destination
// topic chora.governance.policy.violation_detected.v1 is bound to a BINARY
// PROTOCOL_BUFFER schema that rejects the LogEntry a Logging sink publishes,
// and a sink cannot synthesise the mandatory envelope fields. Proven against
// the live schema with a positive control in
// docs/familiar/evidence/2026-08-07/g2-armor-sink.txt §2.
package domain

import "time"

// ArmorLeg names which of the two Armor screening hops produced the verdict.
// The ratified PolicyViolationDetected proto has no leg field, so the leg rides
// in the event description; keeping it a typed value here means the wording is
// derived in exactly one place rather than hand-written per call site.
type ArmorLeg string

const (
	// ArmorLegPre is the PRE-LLM hop screening the user prompt. The blocked
	// text is the caller's input.
	ArmorLegPre ArmorLeg = "PRE"

	// ArmorLegPreContents is the PRE-LLM hop screening the text DERIVED from
	// `contents_json` (G1'-3), as distinct from the flat `prompt` that
	// ArmorLegPre covers. It is a separate leg rather than folded into PRE so
	// governance evidence can tell "the caller's prompt was refused" from "the
	// learner's uploaded artifact was refused": they have different subjects,
	// different remediation, and for now different postures, since this leg is
	// audit-only until promoted.
	ArmorLegPreContents ArmorLeg = "PRE_CONTENTS"

	// ArmorLegPreTools is the PRE-LLM hop screening the text DERIVED from
	// `tools_json` (CHO-2391): the tool NAMES and DESCRIPTIONS the caller
	// supplies, as distinct from the prompt that ArmorLegPre covers and the
	// conversation that ArmorLegPreContents covers. It is a separate leg for
	// the same reason those are: governance evidence must be able to tell "the
	// caller's prompt was refused" from "the caller's TOOL DECLARATIONS were
	// refused". Those have different subjects and different remediation, the
	// second being a caller/BYOA integration problem rather than a learner
	// one, and for now different postures, since this leg is audit-only until
	// promoted.
	ArmorLegPreTools ArmorLeg = "PRE_TOOLS"

	// ArmorLegPost is the POST-LLM hop screening the model completion. The
	// blocked text is model-generated output.
	ArmorLegPost ArmorLeg = "POST"

	// ArmorLegPostToolCalls is the POST-LLM hop screening the text DERIVED from
	// `tool_calls_json` (G1'-2, the tool-call half), as distinct from the
	// completion that ArmorLegPost covers. It is a separate leg for the same
	// reason ArmorLegPreContents is: governance evidence must be able to tell
	// "the model's answer was refused" from "the model's tool ARGUMENTS were
	// refused". Those have different subjects, different remediation, and for
	// now different postures, since this leg is audit-only until promoted.
	ArmorLegPostToolCalls ArmorLeg = "POST_TOOL_CALLS"
)

// PolicyViolationEvent is the domain-shaped payload the gateway hands to the
// ViolationPublisher port when Armor refuses a call. It carries ONLY facts the
// gateway actually holds; the events adapter maps it onto the canonical
// governance.v1.PolicyViolationDetected proto (detector, policy kind, severity
// and resource URI are derived there, the same division of labour the ADR-231
// ExternalEgressAuditEvent uses).
type PolicyViolationEvent struct {
	// ViolationID is a UUIDv7 that doubles as the envelope event_id and the
	// idempotency_key: one blocked request produces one violation identity.
	ViolationID string

	TenantID string
	GCID     string

	// AgentID is the calling agent (e.g. "familiar_companion"). The ratified
	// proto has no agent field; it rides in the description.
	AgentID string

	// InvocationID correlates the violation with the token-usage ledger row
	// emitted for the same refused call.
	InvocationID string

	// Leg is PRE or POST: which hop refused.
	Leg ArmorLeg

	// ArmorTemplate is the full Cloud Model Armor template resource name. It IS
	// the policy that fired, so it maps onto the proto's policy_id.
	ArmorTemplate string

	// Verdict is the Armor verdict that triggered the emission. Only
	// ArmorVerdictBlock reaches the publisher, see Service.emitPolicyViolation.
	Verdict ArmorVerdict

	DetectedAt time.Time

	// W3C trace context, propagated so the governance row correlates with the
	// gateway span that refused the call.
	Traceparent string
	Tracestate  string
}

// Description renders the free-text governance description. The ratified proto
// carries no agent, leg or verdict field and the topic schema is frozen (an
// additive proto field needs a Pub/Sub schema revision and 400s at publish
// until that revision lands), so those three facts ride here as a stable
// key=value tail that a future consumer can parse without guessing.
func (e PolicyViolationEvent) Description() string {
	return "Cloud Model Armor refused the call at the " + string(e.Leg) +
		" leg: agent_id=" + e.AgentID +
		" invocation_id=" + e.InvocationID +
		" verdict=" + e.Verdict.String()
}

// ResourceURI identifies the refused invocation in the URI form the proto
// documents (domain/kind:id).
func (e PolicyViolationEvent) ResourceURI() string {
	return "chora.ai_kernel/invocation:" + e.InvocationID
}
