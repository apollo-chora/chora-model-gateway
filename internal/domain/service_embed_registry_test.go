package domain_test

import (
	"context"
	"errors"
	"testing"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Release B regression: a logical model configured for OpenRouter must be
// dispatched through the OpenAI-compatible adapter against the OpenRouter
// endpoint, and must NEVER invoke the Vertex embedding adapter.
//
// This is the defect the round-10 review found: the Embed flow was hard-wired
// to the Vertex publisher adapter, so the effective embedding route on the Go
// gateway was a PAID Google API even though the registry declared a free
// OpenRouter upstream. The logical id `text-embedding-004` is an alias for the
// deployment's embedding route — it is not evidence about the vendor.
//
// Round 12 tightened the response side: the fake now returns a vector of the
// model's declared length (1024), because a two-element stub proves adapter
// SELECTION and nothing about production vector compatibility.
func TestEmbed_OpenRouterRoute_NeverInvokesVertexAdapter(t *testing.T) {
	openrouter := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: embeddingVector(1024), ModelVersion: "liquid/lfm-2.5-embedding-350m:free"},
	}
	vertex := &fakeEmbedVendor{family: domain.VendorFamilyVertexGemini}

	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		// Exactly the deployment's entry (chora-stack/config/model-gateway/
		// models.prod.yaml): the historical logical id, an OpenRouter
		// upstream, the OpenAI provider, no fallbacks, 1024-dimension output
		// and no `dimensions` parameter support.
		"text-embedding-004": {
			ID:                  "text-embedding-004",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:             "https://openrouter.ai/api/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			APIKeyEnv:           "EMBEDDING_LLM_API_KEY",
			APIKey:              "test-openrouter-key",
		},
	}}

	outbox := &fakeEmbedOutbox{}
	svc := newEmbedServiceWithConfig(t, false, nil, openrouter, outbox, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
		cfg.Embedders = []domain.EmbeddingClient{openrouter, vertex}
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	req.OutputDimensions = 1024

	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)

	// The Vertex adapter was never constructed into the call.
	assert.Equal(t, 0, vertex.calls, "a logical model configured for OpenRouter must never reach the Vertex embedding adapter")

	// The OpenAI-compatible adapter served the RESOLVED route: the registry's
	// upstream model and base URL, not the logical id and not the adapter's
	// construction default.
	require.Equal(t, 1, openrouter.calls)
	assert.Equal(t, "liquid/lfm-2.5-embedding-350m:free", openrouter.lastReq.UpstreamModel,
		"the adapter must send the registry upstream_model, not the logical id")
	assert.Equal(t, "https://openrouter.ai/api/v1", openrouter.lastReq.BaseURL,
		"the adapter must dispatch to the registry base_url")
	assert.Equal(t, domain.LogicalModelID("text-embedding-004"), openrouter.lastReq.LogicalModelID)

	// The requested dimension EQUALS the model's own, so no `dimensions`
	// parameter is forwarded: the approved upstream rejects the parameter, and
	// sending one turns a working route into a 400.
	assert.Equal(t, int32(0), openrouter.lastReq.OutputDimensions,
		"the model declares 1024 dimensions and takes no override; no parameter may be sent")

	// The credential is the resolved entry's own, never the vendor family's
	// platform key.
	assert.Equal(t, "test-openrouter-key", openrouter.lastReq.APIKey)
	assert.Equal(t, "EMBEDDING_LLM_API_KEY", openrouter.lastReq.APIKeyEnv)

	// The response shape is the production one: a 1024-element vector.
	assert.Len(t, resp.Values, 1024)
	assert.Equal(t, "liquid/lfm-2.5-embedding-350m:free", resp.ModelVersion)
	assert.Equal(t, domain.VendorFamilyOpenAI, domain.VendorFamily(resp.Vendor),
		"the ledger and the caller must see the vendor that actually served")

	// The ledger row attributes the route that ran.
	require.Len(t, outbox.events, 1)
	assert.Equal(t, string(domain.VendorFamilyOpenAI), outbox.events[0].Vendor)
	assert.Equal(t, []string{string(domain.VendorFamilyOpenAI) + ":text-embedding-004"}, outbox.events[0].FallbackChain)
}

// embeddingVector builds a deterministic vector of n values.
func embeddingVector(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i%7) / 10
	}
	return v
}

// ----------------------------------------------------------------------------
// Dimension policy (ChatGPT round 12)
// ----------------------------------------------------------------------------

// The model's declared vector length is enforced on the RESPONSE. A 768-value
// vector from a model that produces 1024 must be refused — the caller writes
// it into a vector(1024) column, and a silently short vector would either
// corrupt the embedding or fail later with a message that no longer names the
// model.
func TestEmbed_ReturnedVectorLengthMismatch_IsRefused(t *testing.T) {
	openai := &fakeEmbedVendor{
		family: domain.VendorFamilyOpenAI,
		// The exact defect ChatGPT asked to be rejected: a 768-element vector
		// for the model configured as 1024.
		response: domain.EmbedVendorResponse{Values: embeddingVector(768), ModelVersion: "liquid/lfm-2.5-embedding-350m:free"},
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:                  "text-embedding-004",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:             "https://openrouter.ai/api/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			APIKeyEnv:           "EMBEDDING_LLM_API_KEY",
			APIKey:              "test-key",
		},
	}}
	outbox := &fakeEmbedOutbox{}
	svc := newEmbedServiceWithConfig(t, false, nil, openai, outbox, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	req.OutputDimensions = 1024

	_, err := svc.Embed(context.Background(), req)
	require.Error(t, err, "a wrong-length vector must be refused, never truncated or padded")
	var dimErr *domain.DimensionMismatchError
	require.ErrorAs(t, err, &dimErr)
	assert.Equal(t, int32(1024), dimErr.Expected)
	assert.Equal(t, int32(768), dimErr.Actual)
	assert.Len(t, outbox.events, 0, "a refused vector must not be ledgered")
}

// An explicit dimension the resolved model is known not to produce is refused
// BEFORE dispatch: the model declares 1024 and takes no override, so asking
// for 768 can only produce a vector the caller cannot store.
func TestEmbed_RequestedDimensionTheModelCannotProduce_IsRefused(t *testing.T) {
	openai := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: embeddingVector(1024)},
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:                  "text-embedding-004",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:             "https://openrouter.ai/api/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			APIKeyEnv:           "EMBEDDING_LLM_API_KEY",
			APIKey:              "test-key",
		},
	}}
	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	req.OutputDimensions = 768

	_, err := svc.Embed(context.Background(), req)
	require.Error(t, err)
	var dimErr *domain.DimensionMismatchError
	require.ErrorAs(t, err, &dimErr)
	assert.Equal(t, int32(1024), dimErr.Expected)
	assert.Equal(t, 0, openai.calls, "a request the model cannot honour must never reach the provider")
}

// A model whose entry declares that its upstream ACCEPTS a `dimensions`
// override honours the caller's request, sends the parameter, and holds the
// response to the requested length.
func TestEmbed_OverrideCapableModel_HonoursRequestedDimensions(t *testing.T) {
	openai := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: embeddingVector(256)},
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:                       "text-embedding-004",
			Vendor:                   domain.VendorFamilyOpenAI,
			UpstreamModel:            "text-embedding-3-large",
			BaseURL:                  "https://api.openai.com/v1",
			Capabilities:             []string{"embeddings"},
			EmbeddingDimensions:      3072,
			EmbeddingDimensionsParam: true,
			APIKeyEnv:                "TEXT_LLM_API_KEY",
			APIKey:                   "test-key",
		},
	}}
	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	req.OutputDimensions = 256

	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, int32(256), openai.lastReq.OutputDimensions, "an override-capable model gets the requested dimension")
	assert.Len(t, resp.Values, 256)
}

// An entry that declares nothing about dimensions is dispatched with NO
// parameter (the gateway must not forward one the entry does not declare
// support for), and the RESPONSE is still held to what the caller asked for.
func TestEmbed_UndeclaredDimensions_SendsNoOverrideButValidatesTheResponse(t *testing.T) {
	openai := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: embeddingVector(512)},
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "self-hosted-embed",
			BaseURL:       "https://llm.example.com/v1",
			Capabilities:  []string{"embeddings"},
		},
	}}
	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	req.OutputDimensions = 512

	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, int32(0), openai.lastReq.OutputDimensions,
		"an entry that does not declare `dimensions` support must not be sent the parameter")
	assert.Len(t, resp.Values, 512)

	// The same entry answering with a different length is refused.
	openai.response = domain.EmbedVendorResponse{Values: embeddingVector(768)}
	_, err = svc.Embed(context.Background(), req)
	require.Error(t, err)
	var dimErr *domain.DimensionMismatchError
	require.ErrorAs(t, err, &dimErr)
}

// An empty vector is refused even when the entry declares no dimension: an
// empty embedding must never be ledgered as a successful call.
func TestEmbed_EmptyVector_IsRefused(t *testing.T) {
	openai := &fakeEmbedVendor{family: domain.VendorFamilyOpenAI}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "self-hosted-embed",
			BaseURL:       "https://llm.example.com/v1",
			Capabilities:  []string{"embeddings"},
		},
	}}
	outbox := &fakeEmbedOutbox{}
	svc := newEmbedServiceWithConfig(t, false, nil, openai, outbox, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	_, err := svc.Embed(context.Background(), req)
	require.Error(t, err)
	var dimErr *domain.DimensionMismatchError
	require.ErrorAs(t, err, &dimErr)
	assert.Len(t, outbox.events, 0)
}

// The pinned route requires the approved vector length: the pin supplies it
// (the deployment registry cannot declare its own dimension metadata without
// an edit) and a response of any other length is refused.
func TestEmbed_PinnedRoute_RequiresTheApprovedVectorLength(t *testing.T) {
	openai := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: embeddingVector(1024)},
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:       "https://openrouter.ai/api/v1",
			Capabilities:  []string{"embeddings"},
			APIKeyEnv:     "EMBEDDING_LLM_API_KEY",
			APIKey:        "test-key",
		},
	}}
	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
		cfg.EmbeddingPin = &domain.EmbeddingRoutePin{
			LogicalID:          "text-embedding-004",
			ExpectedDimensions: 1024,
		}
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"

	// Omitted dimension: no override, the pin's 1024 is required and met.
	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)
	assert.Len(t, resp.Values, 1024)
	assert.Equal(t, int32(0), openai.lastReq.OutputDimensions)

	// A 768-element response for the pinned route is refused.
	openai.response = domain.EmbedVendorResponse{Values: embeddingVector(768)}
	_, err = svc.Embed(context.Background(), req)
	require.Error(t, err)
	var dimErr *domain.DimensionMismatchError
	require.ErrorAs(t, err, &dimErr)
	assert.Equal(t, int32(1024), dimErr.Expected)

	// An explicit 768 request against the pinned 1024 route is refused before
	// dispatch.
	req.OutputDimensions = 768
	_, err = svc.Embed(context.Background(), req)
	require.Error(t, err)
	require.ErrorAs(t, err, &dimErr)
	assert.Equal(t, int32(1024), dimErr.Expected)
}

// ----------------------------------------------------------------------------
// Fallback pin (ChatGPT round 12, release blocker)
// ----------------------------------------------------------------------------

// A pinned request must not reach an UNAPPROVED fallback. The pin is a billing
// assertion — the embedding spend goes to the one approved free upstream — so
// while it is wired no fallback is walked at all, not even one whose host and
// upstream happen to look acceptable.
func TestEmbed_PinnedRoute_NeverWalksFallbacks(t *testing.T) {
	paid := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: embeddingVector(1024), ModelVersion: "openai/text-embedding-3-large"},
	}
	primary := &failFirstEmbedVendor{
		fakeEmbedVendor: fakeEmbedVendor{family: domain.VendorFamilyOpenAI},
		failsLeft:       1, // the approved free upstream is down
	}
	// Both targets are openai-family, so ONE adapter serves them: the same
	// adapter object lets the test observe every dispatch. The paid target is
	// the fallback the pin must never reach.
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:                  "text-embedding-004",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:             "https://openrouter.ai/api/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			APIKeyEnv:           "EMBEDDING_LLM_API_KEY",
			APIKey:              "test-key",
			FallbackIDs:         []domain.LogicalModelID{"paid-embed"},
		},
		"paid-embed": {
			ID:                  "paid-embed",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "openai/text-embedding-3-large",
			BaseURL:             "https://api.openai.com/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			APIKeyEnv:           "TEXT_LLM_API_KEY",
			APIKey:              "paid-key",
		},
	}}
	outbox := &fakeEmbedOutbox{}
	svc := newEmbedServiceWithConfig(t, false, nil, primary, outbox, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
		cfg.EmbeddingPin = &domain.EmbeddingRoutePin{
			LogicalID:          "text-embedding-004",
			ExpectedDimensions: 1024,
		}
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"

	_, err := svc.Embed(context.Background(), req)
	require.Error(t, err, "the pinned route is the only route; a failure must surface, not fall back")
	assert.Equal(t, 1, primary.attempts, "the pinned upstream is attempted exactly once")
	assert.Equal(t, 0, paid.calls, "a pinned request must never reach an unapproved fallback")
	require.Len(t, primary.all, 1)
	assert.Equal(t, int32(0), primary.all[0].OutputDimensions,
		"the pinned upstream takes no dimensions parameter")
	assert.Equal(t, "test-key", primary.all[0].APIKey)
	assert.Len(t, outbox.events, 0)
}

// Unpinned, the same registry walks the fallback as before — the pin is what
// disables the walk, not the fallback list itself.
func TestEmbed_UnpinnedRoute_StillWalksFallbacks(t *testing.T) {
	serving := &failFirstEmbedVendor{
		fakeEmbedVendor: fakeEmbedVendor{
			family:   domain.VendorFamilyOpenAI,
			response: domain.EmbedVendorResponse{Values: embeddingVector(1024), ModelVersion: "openai/text-embedding-3-large"},
		},
		failsLeft: 1,
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:                  "text-embedding-004",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:             "https://openrouter.ai/api/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			APIKeyEnv:           "EMBEDDING_LLM_API_KEY",
			APIKey:              "test-key",
			FallbackIDs:         []domain.LogicalModelID{"paid-embed"},
		},
		"paid-embed": {
			ID:                  "paid-embed",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "openai/text-embedding-3-large",
			BaseURL:             "https://api.openai.com/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			APIKeyEnv:           "TEXT_LLM_API_KEY",
			APIKey:              "paid-key",
		},
	}}
	svc := newEmbedServiceWithConfig(t, false, nil, serving, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"

	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, 2, serving.attempts)
	assert.Len(t, resp.Values, 1024)
}

// ----------------------------------------------------------------------------
// Credential isolation (ChatGPT round 12)
// ----------------------------------------------------------------------------

// Each target carries ITS OWN resolved credential. A request routed to a
// different host must never carry another provider's key: the two entries here
// name different credentials AND different endpoints, and the walk from the
// failing primary to the fallback must re-point both.
func TestEmbed_EachTargetCarriesItsOwnCredentialAndEndpoint(t *testing.T) {
	serving := &failFirstEmbedVendor{
		fakeEmbedVendor: fakeEmbedVendor{
			family:   domain.VendorFamilyOpenAI,
			response: domain.EmbedVendorResponse{Values: embeddingVector(1024), ModelVersion: "openai/text-embedding-3-large"},
		},
		failsLeft: 1,
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:                  "text-embedding-004",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:             "https://openrouter.ai/api/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			APIKeyEnv:           "EMBEDDING_LLM_API_KEY",
			APIKey:              "openrouter-key",
			FallbackIDs:         []domain.LogicalModelID{"openai-embed"},
		},
		"openai-embed": {
			ID:                  "openai-embed",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "openai/text-embedding-3-large",
			BaseURL:             "https://api.openai.com/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			APIKeyEnv:           "TEXT_LLM_API_KEY",
			APIKey:              "openai-key",
		},
	}}
	svc := newEmbedServiceWithConfig(t, false, nil, serving, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"

	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)
	require.Equal(t, 2, len(serving.all), "the primary is attempted, then the fallback")

	// The primary dispatch carried the OpenRouter entry's credential...
	assert.Equal(t, "openrouter-key", serving.all[0].APIKey)
	assert.Equal(t, "EMBEDDING_LLM_API_KEY", serving.all[0].APIKeyEnv)
	assert.Equal(t, "https://openrouter.ai/api/v1", serving.all[0].BaseURL)

	// ...and the fallback carried ITS OWN, never the primary's.
	assert.Equal(t, "openai-key", serving.all[1].APIKey)
	assert.Equal(t, "TEXT_LLM_API_KEY", serving.all[1].APIKeyEnv)
	assert.Equal(t, "https://api.openai.com/v1", serving.all[1].BaseURL)
	assert.Len(t, resp.Values, 1024)
}

// An entry with NO credential reference dispatched to a provider that
// authenticates every request is refused. This is the hole ChatGPT found: the
// old rule only refused "a reference that resolves to empty", so an entry with
// both fields empty went out anonymously.
func TestEmbed_NoCredentialReferenceOnAuthenticatedHost_IsRefused(t *testing.T) {
	openai := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: embeddingVector(1024)},
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:                  "text-embedding-004",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:             "https://openrouter.ai/api/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
			// Both empty — the case the old check let through.
		},
	}}
	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	_, err := svc.Embed(context.Background(), req)
	require.Error(t, err)
	var credErr *domain.CredentialError
	require.ErrorAs(t, err, &credErr)
	assert.Contains(t, err.Error(), "openrouter.ai")
	assert.Equal(t, 0, openai.calls, "an anonymously-called authenticated host must never be dispatched to")
}

// The pinned route requires a configured credential reference even when the
// host table would not: the approved OpenRouter endpoint is never reached
// anonymously, and the refusal names the reference, never its value.
func TestEmbed_PinnedRoute_RequiresAConfiguredCredential(t *testing.T) {
	openai := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: embeddingVector(1024)},
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "liquid/lfm-2.5-embedding-350m:free",
			// A host the credential table does not know: only the pin's own
			// requirement applies.
			BaseURL:             "https://embeddings.internal.example.com/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 1024,
		},
	}}
	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
		cfg.EmbeddingPin = &domain.EmbeddingRoutePin{
			LogicalID:          "text-embedding-004",
			ExpectedDimensions: 1024,
		}
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	_, err := svc.Embed(context.Background(), req)
	require.Error(t, err)
	var credErr *domain.CredentialError
	require.ErrorAs(t, err, &credErr)
	assert.Equal(t, 0, openai.calls)
}

// A public / self-hosted OpenAI-compatible endpoint legitimately needs no
// credential: the rule is provider-specific, so such an entry is dispatched
// with no key at all.
func TestEmbed_PublicEndpointWithoutCredential_IsAllowed(t *testing.T) {
	openai := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: embeddingVector(768)},
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:                  "text-embedding-004",
			Vendor:              domain.VendorFamilyOpenAI,
			UpstreamModel:       "self-hosted-embed",
			BaseURL:             "https://llm.example.com/v1",
			Capabilities:        []string{"embeddings"},
			EmbeddingDimensions: 768,
		},
	}}
	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)
	assert.Len(t, resp.Values, 768)
	assert.Empty(t, openai.lastReq.APIKey)
	assert.Empty(t, openai.lastReq.APIKeyEnv)
}

// The same logical id resolves to whatever the registry currently declares.
// Re-pointing the entry at a different provider must change the adapter that
// answers — the id's name plays no part in the decision.
func TestEmbed_RegistryRePoint_ChangesTheServingAdapter(t *testing.T) {
	openai := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: []float32{0.1}, ModelVersion: "openai/text-embedding-3-small"},
	}
	vertex := &fakeEmbedVendor{
		family:   domain.VendorFamilyVertexGemini,
		response: domain.EmbedVendorResponse{Values: []float32{0.9}, ModelVersion: "text-embedding-004"},
	}

	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyVertexGemini,
			UpstreamModel: "text-embedding-004",
			BaseURL:       "https://asia-southeast1-aiplatform.googleapis.com",
			Capabilities:  []string{"embeddings"},
		},
	}}

	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
		cfg.Embedders = []domain.EmbeddingClient{openai, vertex}
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)

	assert.Equal(t, 0, openai.calls)
	assert.Equal(t, 1, vertex.calls, "a registry entry that declares the Vertex provider must dispatch to the Vertex adapter")
	assert.Equal(t, []float32{0.9}, resp.Values)
}

// An entry whose provider has no wired embedding adapter is refused, never
// re-routed to an adapter that happens to be constructed.
func TestEmbed_ProviderWithoutWiredAdapter_IsRefused(t *testing.T) {
	openai := &fakeEmbedVendor{family: domain.VendorFamilyOpenAI, response: domain.EmbedVendorResponse{Values: []float32{0.1}}}

	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		// anthropic is a known provider with no embedding adapter. Its entry
		// declares a credential, so the refusal under test is the MISSING
		// ADAPTER, not the credential policy.
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyAnthropic,
			UpstreamModel: "text-embedding-004",
			BaseURL:       "https://api.anthropic.com",
			Capabilities:  []string{"embeddings"},
			APIKeyEnv:     "ANTHROPIC_API_KEY",
			APIKey:        "test-key",
		},
	}}

	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
		cfg.Embedders = []domain.EmbeddingClient{openai}
	})

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	require.Error(t, err)
	var noProvider *domain.NoProviderError
	require.ErrorAs(t, err, &noProvider)
	assert.Equal(t, 0, openai.calls, "an unroutable provider must not be served by a constructed adapter")
}

// A logical id that is not in the registry at all is refused before any
// adapter is consulted. The id is an allowlisted embedding id so the refusal
// comes from the registry, not the name-shape guard.
func TestEmbed_UnknownLogicalModel_IsRefused(t *testing.T) {
	openai := &fakeEmbedVendor{family: domain.VendorFamilyOpenAI, response: domain.EmbedVendorResponse{Values: []float32{0.1}}}

	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{}}
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-005"
	_, err := svc.Embed(context.Background(), req)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not in the registry")
	assert.Equal(t, 0, openai.calls)
}

// A registry entry that does not advertise the embeddings capability is
// refused even when its provider HAS a wired embedding adapter.
func TestEmbed_NonEmbeddingCapability_IsRefused(t *testing.T) {
	openai := &fakeEmbedVendor{family: domain.VendorFamilyOpenAI, response: domain.EmbedVendorResponse{Values: []float32{0.1}}}

	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "gpt-4o",
			BaseURL:       "https://api.openai.com/v1",
			Capabilities:  []string{"chat"},
		},
	}}

	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	require.Error(t, err)
	var capErr *domain.CapabilityError
	require.ErrorAs(t, err, &capErr)
	assert.Equal(t, 0, openai.calls)
}

// An entry with no resolvable endpoint is refused rather than dispatched to
// whatever the adapter happens to be constructed with — the "silent paid
// route" hole.
func TestEmbed_NoResolvableBaseURL_IsRefused(t *testing.T) {
	openai := &fakeEmbedVendor{family: domain.VendorFamilyOpenAI, response: domain.EmbedVendorResponse{Values: []float32{0.1}}}

	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:       "",
			Capabilities:  []string{"embeddings"},
			APIKeyEnv:     "EMBEDDING_LLM_API_KEY",
			APIKey:        "test-key",
		},
	}}

	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "base_url")
	assert.Equal(t, 0, openai.calls)
}

// A declared credential reference that resolves to empty is refused before
// the call goes out unauthenticated.
func TestEmbed_EmptyCredential_IsRefused(t *testing.T) {
	openai := &fakeEmbedVendor{family: domain.VendorFamilyOpenAI, response: domain.EmbedVendorResponse{Values: []float32{0.1}}}

	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:       "https://openrouter.ai/api/v1",
			Capabilities:  []string{"embeddings"},
			APIKeyEnv:     "EMBEDDING_LLM_API_KEY",
			APIKey:        "",
		},
	}}

	svc := newEmbedServiceWithConfig(t, false, nil, openai, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	require.Error(t, err)
	var credErr *domain.CredentialError
	require.ErrorAs(t, err, &credErr)
	assert.Equal(t, 0, openai.calls)
}

// failFirstEmbedVendor fails the first EmbedText call and answers the rest,
// so a test can drive a primary→fallback walk through one family-keyed
// adapter. Every request is recorded, so a test can prove each target carried
// its OWN resolved route and credential.
type failFirstEmbedVendor struct {
	fakeEmbedVendor
	failsLeft int
	attempts  int
	all       []domain.EmbedVendorRequest
}

func (f *failFirstEmbedVendor) EmbedText(ctx context.Context, req domain.EmbedVendorRequest) (domain.EmbedVendorResponse, error) {
	f.attempts++
	f.all = append(f.all, req)
	if f.failsLeft > 0 {
		f.failsLeft--
		return domain.EmbedVendorResponse{}, errors.New("openrouter: upstream timeout")
	}
	return f.fakeEmbedVendor.EmbedText(ctx, req)
}

// A registry-declared fallback is walked only through the same resolution +
// policy checks: a fallback that cannot serve the request is skipped, and one
// that can is dispatched with ITS OWN resolved route. Both targets are
// OpenAI-provider entries, so the single wired OpenAI-family adapter serves
// both — which is exactly what makes the per-target upstream observable.
func TestEmbed_FallbackChain_ResolvedWithTheSameChecks(t *testing.T) {
	serving := &failFirstEmbedVendor{
		fakeEmbedVendor: fakeEmbedVendor{
			family:   domain.VendorFamilyOpenAI,
			response: domain.EmbedVendorResponse{Values: []float32{0.3}, ModelVersion: "openai/text-embedding-3-small"},
		},
		failsLeft: 1,
	}

	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:       "https://openrouter.ai/api/v1",
			Capabilities:  []string{"embeddings"},
			APIKeyEnv:     "EMBEDDING_LLM_API_KEY",
			APIKey:        "test-key",
			FallbackIDs:   []domain.LogicalModelID{"openai-embed"},
		},
		"openai-embed": {
			ID:            "openai-embed",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "openai/text-embedding-3-small",
			BaseURL:       "https://api.openai.com/v1",
			Capabilities:  []string{"embeddings"},
			APIKeyEnv:     "EMBEDDING_LLM_API_KEY",
			APIKey:        "test-key",
		},
	}}

	svc := newEmbedServiceWithConfig(t, false, nil, serving, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	req := validEmbedRequest()
	req.LogicalModelID = "text-embedding-004"
	resp, err := svc.Embed(context.Background(), req)
	require.NoError(t, err)

	assert.Equal(t, 2, serving.attempts, "the failing primary is attempted, then the resolved fallback")
	assert.Equal(t, "openai/text-embedding-3-small", serving.lastReq.UpstreamModel,
		"the fallback dispatches with its OWN resolved upstream, not the primary's")
	assert.Equal(t, "https://api.openai.com/v1", serving.lastReq.BaseURL)
	assert.Equal(t, []float32{0.3}, resp.Values)
}

// A fallback that does not advertise the embeddings capability is skipped,
// not dispatched to.
func TestEmbed_FallbackWithoutEmbeddingCapability_IsSkipped(t *testing.T) {
	serving := &failFirstEmbedVendor{
		fakeEmbedVendor: fakeEmbedVendor{family: domain.VendorFamilyOpenAI},
		failsLeft:       1,
	}
	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:       "https://openrouter.ai/api/v1",
			Capabilities:  []string{"embeddings"},
			APIKeyEnv:     "EMBEDDING_LLM_API_KEY",
			APIKey:        "test-key",
			FallbackIDs:   []domain.LogicalModelID{"chat-only"},
		},
		"chat-only": {
			ID:            "chat-only",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "gpt-4o",
			BaseURL:       "https://api.openai.com/v1",
			Capabilities:  []string{"chat"},
		},
	}}

	svc := newEmbedServiceWithConfig(t, false, nil, serving, &fakeEmbedOutbox{}, func(cfg *domain.ServiceConfig) {
		cfg.Models = reg
	})

	_, err := svc.Embed(context.Background(), validEmbedRequest())
	require.Error(t, err)
	assert.Equal(t, 1, serving.attempts, "the primary is attempted once; the unusable fallback is never dispatched to")
	var capErr *domain.CapabilityError
	assert.ErrorAs(t, err, &capErr)
}
