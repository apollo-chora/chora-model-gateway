package gemini

import (
	"context"
	"fmt"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// Provider adapts Client to the domain.CapabilityProvider canonical interface
// (internal/domain/provider.go). It is a thin wrapper: it translates the
// canonical request to the VendorRequest shape the Client already speaks,
// and translates the VendorResponse back to the canonical response.
//
// The Client keeps its VendorClient / GroundedVendorClient implementations
// for the governance layer (Service.Invoke / Service.GroundedSearch); this
// wrapper exposes the same dispatch through the canonical Provider contract
// so the Python port's single-dispatch-surface shape is preserved.
type Provider struct {
	client *Client
}

// NewProvider constructs a Provider wrapping the given Client. The Client
// must already be constructed via New().
func NewProvider(client *Client) *Provider {
	return &Provider{client: client}
}

// Generate implements domain.TextGenerator.
func (p *Provider) Generate(ctx context.Context, req domain.GenerateRequest) (*domain.GenerateResponse, error) {
	vendorReq := domain.VendorRequest{
		Vendor:           domain.VendorFamilyVertexGemini,
		LogicalModelID:   domain.LogicalModelID(req.Model),
		Prompt:           req.Prompt,
		SystemPrompt:     req.SystemPrompt,
		ContentsJSON:     req.ContentsJSON,
		ToolsJSON:        req.ToolsJSON,
		GenerationConfig: req.GenerationConfig,
	}
	if req.Grounding != nil && req.Grounding.Enabled {
		vendorReq.ResponseModality = "GROUNDED"
	}
	vendorResp, err := p.client.Generate(ctx, vendorReq)
	if err != nil {
		return nil, err
	}
	return &domain.GenerateResponse{
		Text:          vendorResp.Completion,
		Citations:     vendorResp.Citations,
		SearchQueries: vendorResp.SearchQueries,
		FinishReason:  vendorResp.FinishReason,
		Usage:         toCanonicalUsage(vendorResp.Usage),
		Vendor:        string(domain.VendorFamilyVertexGemini),
		ModelVersion:  vendorResp.ModelVersion,
		RevisedPrompt: vendorResp.RevisedPrompt,
		ToolCallsJSON: vendorResp.ToolCallsJSON,
	}, nil
}

// Embed implements domain.Embedder. Gemini's :generateContent surface does
// not serve embeddings; the gateway routes embeddings through the dedicated
// vertexembed adapter. This method returns a clear not-supported error so
// a caller can distinguish "this vendor has no embedding surface" from a
// transport failure.
func (p *Provider) Embed(ctx context.Context, req domain.EmbedRequest) (*domain.EmbedResponse, error) {
	return nil, fmt.Errorf("gemini: embeddings not supported via this adapter; use vertexembed")
}

// GenerateImage implements domain.ImageGenerator. Image generation rides
// the same :generateContent call with responseModalities=["IMAGE"].
func (p *Provider) GenerateImage(ctx context.Context, req domain.ImageRequest) (*domain.ImageResponse, error) {
	vendorResp, err := p.client.Generate(ctx, domain.VendorRequest{
		Vendor:           domain.VendorFamilyVertexGemini,
		LogicalModelID:   domain.LogicalModelID(req.Model),
		Prompt:           req.Prompt,
		ResponseModality: "IMAGE",
		GenerationConfig: req.GenerationConfig,
	})
	if err != nil {
		return nil, err
	}
	return &domain.ImageResponse{
		ImageBytes:    vendorResp.ImageBytes,
		ImageMIMEType: vendorResp.ImageMIMEType,
		RevisedPrompt: vendorResp.RevisedPrompt,
		Usage:         toCanonicalUsage(vendorResp.Usage),
		ModelVersion:  vendorResp.ModelVersion,
	}, nil
}

// GroundedGenerate implements domain.GroundedGenerator.
func (p *Provider) GroundedGenerate(ctx context.Context, req domain.GroundedGenerateRequest) (*domain.GroundedGenerateResponse, error) {
	vendorResp, err := p.client.GroundedGenerate(ctx, domain.GroundedVendorRequest{
		LogicalModelID: domain.LogicalModelID(req.Model),
		Directive:       req.Prompt,
		SystemPrompt:    req.SystemPrompt,
		MaxResults:      req.MaxResults,
	})
	if err != nil {
		return nil, err
	}
	return &domain.GroundedGenerateResponse{
		Text:          vendorResp.Answer,
		Citations:     vendorResp.Citations,
		SearchQueries: vendorResp.WebSearchQueries,
		Usage:         toCanonicalUsage(vendorResp.Usage),
		ModelVersion:  vendorResp.ModelVersion,
		FinishReason:  vendorResp.FinishReason,
	}, nil
}

// toCanonicalUsage maps the governance-layer TokenUsage to the canonical
// TokenUsage struct. The governance layer tracks input / output / cached /
// cost; the canonical struct adds the extended fields (reasoning, image units,
// embedding units, grounding units) that stay zero on this adapter.
func toCanonicalUsage(u domain.TokenUsage) domain.TokenUsage {
	return domain.TokenUsage{
		InputTokens:       u.InputTokens,
		CachedTokens:      u.CachedTokens,
		OutputTokens:      u.OutputTokens,
		CostMicros:        u.CostMicros,
	}
}
