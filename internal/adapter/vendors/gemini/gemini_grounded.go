package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// GroundedGenerate implements domain.GroundedVendorClient against Vertex AI
// "Grounding with Google Search" (ADR-231 D1). It rides the SAME
// :generateContent call as Generate, injecting the google_search built-in tool
// and parsing the structured citations out of candidates[].groundingMetadata.
//
// The redirect URI on each citation is EPHEMERAL (~30-day expiry, ADR-231 D4);
// the durable render field is Domain. The snippet is best-effort — mapped from
// groundingSupports segment text (may be empty). The searchEntryPoint HTML is a
// Google display obligation the Far Sight FE must render (D5).
func (c *Client) GroundedGenerate(ctx context.Context, req domain.GroundedVendorRequest) (domain.GroundedVendorResponse, error) {
	if err := domain.CheckContextCancellation(ctx); err != nil {
		return domain.GroundedVendorResponse{}, err
	}
	model := string(req.LogicalModelID)
	url := c.groundedURL(model)

	body := buildGroundedBody(req)
	rawBody, err := json.Marshal(body)
	if err != nil {
		return domain.GroundedVendorResponse{}, fmt.Errorf("gemini grounded: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(rawBody))
	if err != nil {
		return domain.GroundedVendorResponse{}, fmt.Errorf("gemini grounded: build request: %w", err)
	}
	token, err := c.tokens.Token(ctx)
	if err != nil {
		return domain.GroundedVendorResponse{}, fmt.Errorf("gemini grounded: token: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+token)
	httpReq.Header.Set("Content-Type", "application/json")
	if req.Traceparent != "" {
		httpReq.Header.Set("traceparent", req.Traceparent)
	}
	if req.Tracestate != "" {
		httpReq.Header.Set("tracestate", req.Tracestate)
	}

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return domain.GroundedVendorResponse{}, fmt.Errorf("gemini grounded: HTTP do: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		return domain.GroundedVendorResponse{}, &domain.ProviderError{
			Status:   httpResp.StatusCode,
			Endpoint: httpReq.URL.String(),
			Body:     strings.TrimSpace(string(respBody)),
		}
	}

	var parsed geminiGroundedResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return domain.GroundedVendorResponse{}, fmt.Errorf("gemini grounded: unmarshal response: %w", err)
	}
	return parsed.toDomain(model, req.MaxResults), nil
}

// groundedURL builds the :generateContent URL, mirroring Generate's regional /
// global host resolution (ADR-163 + feedback_model_armor_regional_endpoint).
func (c *Client) groundedURL(model string) string {
	host := c.endpoint
	if host == "" {
		if c.location == "global" {
			host = "https://aiplatform.googleapis.com"
		} else {
			host = fmt.Sprintf("https://%s-aiplatform.googleapis.com", c.location)
		}
	}
	return fmt.Sprintf(
		"%s/v1/projects/%s/locations/%s/publishers/google/models/%s:generateContent",
		host, c.project, c.location, model,
	)
}

// groundedSearchTool is the Vertex request fragment that enables "Grounding
// with Google Search" — the built-in google_search tool. Reuses the existing
// geminiBody.Tools json.RawMessage passthrough (ADR-231 D1: no new client).
var groundedSearchTool = json.RawMessage(`[{"google_search":{}}]`)

func buildGroundedBody(req domain.GroundedVendorRequest) geminiBody {
	body := geminiBody{
		Contents: []geminiContent{
			{Role: "user", Parts: []geminiPart{{Text: req.Directive}}},
		},
		Tools: groundedSearchTool,
	}
	if req.SystemPrompt != "" {
		body.SystemInstruction = &geminiContent{Parts: []geminiPart{{Text: req.SystemPrompt}}}
	}
	return body
}

// ---------------------------------------------------------------------------
// Grounded wire shapes — Vertex generateContent + groundingMetadata subset.
// ---------------------------------------------------------------------------

type geminiGroundedResponse struct {
	Candidates []struct {
		Content           geminiContent            `json:"content"`
		FinishReason      string                   `json:"finishReason"`
		GroundingMetadata *geminiGroundingMetadata `json:"groundingMetadata"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount        int64 `json:"promptTokenCount"`
		CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
		CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
}

type geminiGroundingMetadata struct {
	WebSearchQueries []string `json:"webSearchQueries"`
	SearchEntryPoint *struct {
		RenderedContent string `json:"renderedContent"`
	} `json:"searchEntryPoint"`
	GroundingChunks []struct {
		Web *struct {
			URI    string `json:"uri"`
			Title  string `json:"title"`
			Domain string `json:"domain"`
		} `json:"web"`
	} `json:"groundingChunks"`
	GroundingSupports []struct {
		Segment *struct {
			Text string `json:"text"`
		} `json:"segment"`
		GroundingChunkIndices []int     `json:"groundingChunkIndices"`
		ConfidenceScores      []float64 `json:"confidenceScores"`
	} `json:"groundingSupports"`
}

// groundedSurchargeMicros is the flat per-grounded-prompt Google Search
// grounding fee (~$0.035/prompt on Gemini 2.5, ADR-231 D1) added on top of the
// token cost. Phase-4 pricing-config follow-up makes this per-model/tunable.
const groundedSurchargeMicros int64 = 35_000

func (g geminiGroundedResponse) toDomain(fallbackModel string, maxResults int32) domain.GroundedVendorResponse {
	resp := domain.GroundedVendorResponse{
		ModelVersion: g.ModelVersion,
		Usage: domain.TokenUsage{
			InputTokens:  g.UsageMetadata.PromptTokenCount,
			OutputTokens: g.UsageMetadata.CandidatesTokenCount,
			CachedTokens: g.UsageMetadata.CachedContentTokenCount,
			CostMicros: geminiCostMicros(fallbackModel, g.UsageMetadata.PromptTokenCount, g.UsageMetadata.CandidatesTokenCount, g.UsageMetadata.CachedContentTokenCount) +
				groundedSurchargeMicros,
		},
		FinishReason: domain.FinishReasonUnspecified,
	}
	if resp.ModelVersion == "" {
		resp.ModelVersion = fallbackModel
	}
	if len(g.Candidates) == 0 {
		return resp
	}
	cand := g.Candidates[0]
	resp.FinishReason = mapGeminiFinishReason(cand.FinishReason)

	var sb strings.Builder
	for _, p := range cand.Content.Parts {
		sb.WriteString(p.Text)
	}
	resp.Answer = sb.String()

	if cand.GroundingMetadata == nil {
		return resp
	}
	gm := cand.GroundingMetadata
	resp.WebSearchQueries = gm.WebSearchQueries
	if gm.SearchEntryPoint != nil {
		resp.SearchEntryPointHTML = gm.SearchEntryPoint.RenderedContent
	}

	// Chunks → citations (index-aligned so groundingSupports can attach snippets).
	citations := make([]domain.GroundedCitation, 0, len(gm.GroundingChunks))
	for _, ch := range gm.GroundingChunks {
		if ch.Web == nil {
			citations = append(citations, domain.GroundedCitation{})
			continue
		}
		citations = append(citations, domain.GroundedCitation{
			URI:    ch.Web.URI,
			Title:  ch.Web.Title,
			Domain: ch.Web.Domain,
		})
	}
	// Supports → snippet + confidence, mapped onto the chunk(s) each backs.
	for _, sup := range gm.GroundingSupports {
		var text string
		if sup.Segment != nil {
			text = sup.Segment.Text
		}
		for j, idx := range sup.GroundingChunkIndices {
			if idx < 0 || idx >= len(citations) {
				continue
			}
			if citations[idx].Snippet == "" {
				citations[idx].Snippet = text
			}
			if j < len(sup.ConfidenceScores) && citations[idx].Confidence == 0 {
				citations[idx].Confidence = float32(sup.ConfidenceScores[j])
			}
		}
	}

	// Drop malformed (no uri/title) chunks then apply the caller's citation cap.
	renderable := citations[:0]
	for _, c := range citations {
		if c.URI == "" && c.Title == "" {
			continue
		}
		renderable = append(renderable, c)
	}
	if maxResults > 0 && int32(len(renderable)) > maxResults {
		renderable = renderable[:maxResults]
	}
	resp.Citations = renderable
	return resp
}
