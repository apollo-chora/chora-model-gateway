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
func TestEmbed_OpenRouterRoute_NeverInvokesVertexAdapter(t *testing.T) {
	openrouter := &fakeEmbedVendor{
		family:   domain.VendorFamilyOpenAI,
		response: domain.EmbedVendorResponse{Values: []float32{0.1, 0.2}, ModelVersion: "liquid/lfm-2.5-embedding-350m:free"},
	}
	vertex := &fakeEmbedVendor{family: domain.VendorFamilyVertexGemini}

	reg := &mapModelResolver{infos: map[domain.LogicalModelID]domain.ModelInfo{
		// Exactly the deployment's entry (chora-stack/config/model-gateway/
		// models.prod.yaml): the historical logical id, an OpenRouter
		// upstream, the OpenAI provider, no fallbacks.
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyOpenAI,
			UpstreamModel: "liquid/lfm-2.5-embedding-350m:free",
			BaseURL:       "https://openrouter.ai/api/v1",
			Capabilities:  []string{"embeddings"},
			APIKeyEnv:     "EMBEDDING_LLM_API_KEY",
			APIKey:        "test-openrouter-key",
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

	// The response shape and the requested dimensions are preserved.
	assert.Len(t, resp.Values, 2)
	assert.Equal(t, "liquid/lfm-2.5-embedding-350m:free", resp.ModelVersion)
	assert.Equal(t, domain.VendorFamilyOpenAI, domain.VendorFamily(resp.Vendor),
		"the ledger and the caller must see the vendor that actually served")

	// The ledger row attributes the route that ran.
	require.Len(t, outbox.events, 1)
	assert.Equal(t, string(domain.VendorFamilyOpenAI), outbox.events[0].Vendor)
	assert.Equal(t, []string{string(domain.VendorFamilyOpenAI) + ":text-embedding-004"}, outbox.events[0].FallbackChain)
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
		// anthropic is a known provider with no embedding adapter.
		"text-embedding-004": {
			ID:            "text-embedding-004",
			Vendor:        domain.VendorFamilyAnthropic,
			UpstreamModel: "text-embedding-004",
			BaseURL:       "https://api.anthropic.com",
			Capabilities:  []string{"embeddings"},
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
// adapter.
type failFirstEmbedVendor struct {
	fakeEmbedVendor
	failsLeft int
	attempts  int
}

func (f *failFirstEmbedVendor) EmbedText(ctx context.Context, req domain.EmbedVendorRequest) (domain.EmbedVendorResponse, error) {
	f.attempts++
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
