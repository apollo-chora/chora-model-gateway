package domain

// AgentPolicy is the per-agent routing + guardrail decision the policy
// loader resolves at every Invoke. Loaded from a YAML file shipped via
// Secret Manager / GCS at gateway boot (per ADR-163 §Code consequences 3).
//
// The PolicyLoader port returns this struct given an agent_id +
// logical_model_id from the InvokeRequest.
type AgentPolicy struct {
	// Logical agent identifier (matches chora-infra/agents-cli/registry.json
	// crew names like "qgen_question" or graph-node like
	// "ai_assist_crew.broker_call").
	AgentID string

	// Resolved canonical logical model id. May differ from the caller's
	// request when the policy enforces a tier (e.g. all qgen_question
	// requests use gemini-2.5-pro regardless of caller hint).
	ResolvedLogicalModelID LogicalModelID

	// Vendor family the gateway should dispatch to.
	Vendor VendorFamily

	// Ordered fallback chain of (vendor:logical_model) pairs to try on
	// vendor 5xx or rate-limit. Each entry uses the same VendorFamily
	// enum semantics. Empty = no fallback (single-shot).
	FallbackChain []AgentPolicyFallback

	// ArmorTemplate is the fully-qualified Cloud Model Armor template
	// resource name the gateway calls Sanitize{User,Model}* against.
	// Format: projects/.../locations/.../templates/...
	// Empty string = no Model Armor screening (only allowed for the
	// `permissive` tier in `agent-guardrail-mapping.yaml`).
	ArmorTemplate string
}

// AgentPolicyFallback is one entry in the routing fallback chain.
type AgentPolicyFallback struct {
	Vendor                 VendorFamily
	ResolvedLogicalModelID LogicalModelID
}
