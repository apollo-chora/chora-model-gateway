package domain

import (
	"errors"
	"fmt"
)

// Output modality selectors carried on InvokeRequest.ResponseModality.
// The gRPC surface passes the strings through verbatim; these constants
// exist so the domain and the HTTP adapter agree on the comparison.
const (
	ModalityText  = "TEXT"
	ModalityImage = "IMAGE"
	// ModalityGrounded routes the call through the provider's hosted
	// web-search surface. Unlike TEXT and IMAGE, this needs a different
	// ENDPOINT as well as a different body shape, because every provider
	// puts its search tool somewhere different: OpenAI on /v1/responses,
	// Anthropic on /v1/messages.
	ModalityGrounded = "GROUNDED"
)

// Capability names a registry entry may advertise. A caller addressing a
// surface the model does not claim is refused rather than silently
// downgraded to text.
const (
	CapabilityChat       = "chat"
	CapabilityTools      = "tools"
	CapabilityVision     = "vision"
	CapabilityImage      = "image"
	CapabilityEmbeddings = "embeddings"
	// CapabilityWebSearch means the entry's provider offers a HOSTED web
	// search that runs server-side during the call. It is opt-in per registry
	// entry: most hosted models do not have it, and a model that does not
	// advertise it is refused rather than dispatched and ignored.
	CapabilityWebSearch = "web_search"
)

// KnownCapabilities is the vocabulary the registry loader checks against, so a
// typo in a registry file surfaces at boot instead of silently disabling a
// gate.
var KnownCapabilities = []string{
	CapabilityChat,
	CapabilityTools,
	CapabilityVision,
	CapabilityImage,
	CapabilityEmbeddings,
	CapabilityWebSearch,
}

// modalityCapability maps each non-text modality to the capability a target
// must advertise to serve it. A modality absent from this map requires only
// the chat capability.
//
// It is a map rather than a switch so adding a modality cannot silently skip
// the capability gate — the compiler will point at every site that needs a new
// entry.
var modalityCapability = map[string]string{
	ModalityImage:    CapabilityImage,
	ModalityGrounded: CapabilityWebSearch,
}

// RequiredCapability returns the capability a target must advertise to serve
// the given modality. Text needs only the chat capability.
func RequiredCapability(modality string) string {
	if cap, ok := modalityCapability[modality]; ok {
		return cap
	}
	return CapabilityChat
}

// Validate checks the request envelope. A missing tenant or model is a
// caller bug, and discovering it at the vendor would mean a billable
// request with no attribution — so it is refused before any spend.
func (r InvokeRequest) Validate() error {
	if r.TenantID == "" {
		return errors.New("tenant_id required")
	}
	if r.GCID == "" {
		return errors.New("gcid required")
	}
	if r.AgentID == "" {
		return errors.New("agent_id required")
	}
	if r.LogicalModelID == "" {
		return errors.New("logical_model_id required")
	}
	switch r.ResponseModality {
	case "", ModalityText, ModalityImage, ModalityGrounded:
	default:
		return fmt.Errorf("response_modality %q is not one of TEXT, IMAGE, GROUNDED", r.ResponseModality)
	}
	// A tool-calling turn carries its conversation in contents_json; a
	// text-only turn carries it in prompt. Requiring at least one keeps a
	// caller from dispatching an empty request that a provider will bill for.
	if r.Prompt == "" && r.ContentsJSON == "" {
		return errors.New("prompt or contents_json required (an empty request is still billed by most providers)")
	}
	return nil
}

// numberFrom coerces a generation-config value to a float. JSON decoding
// yields float64 for every number, but a hand-built map from the HTTP
// adapter may carry an int, so both are accepted.
func numberFrom(v any) (float64, bool) {
	switch n := v.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int32:
		return float64(n), true
	case int64:
		return float64(n), true
	default:
		return 0, false
	}
}
