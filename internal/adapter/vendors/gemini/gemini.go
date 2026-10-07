// Package gemini implements domain.VendorClient against Vertex AI Gemini
// via the regional HTTP REST API.
//
// Per ADR-163 + feedback_model_armor_regional_endpoint, the endpoint MUST
// be pinned regionally:
//
//	asia-southeast1-aiplatform.googleapis.com
//
// We use raw HTTP + JSON rather than the cloud.google.com/go/vertexai SDK
// because:
//  1. The SDK does not let us cleanly inject a custom http.RoundTripper
//     for mocked-vendor unit tests.
//  2. The gateway dispatches one provider-neutral request shape to each
//     vendor; coupling tightly to a vendor SDK costs more than the JSON
//     shape adds.
//  3. Per ADR-163 the gateway is the ONLY caller of Vertex AI Gemini —
//     there's no second-consumer benefit from an SDK abstraction.
package gemini

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/5007-Capstone/chora/services/chora-model-gateway/internal/domain"
)

// TokenProvider returns a short-lived OAuth2 access token. Production wiring
// passes a Google Workload-Identity-Federation token source; tests pass a
// stub returning a fixed string.
type TokenProvider interface {
	Token(ctx context.Context) (string, error)
}

// Client is the Gemini VendorClient. Construct via New(); pass the resulting
// pointer to domain.NewService as one of cfg.Vendors.
type Client struct {
	httpClient *http.Client
	tokens     TokenProvider
	project    string
	location   string
	// endpoint allows tests to override the API host. Production callers
	// leave it empty and the client constructs
	// https://{location}-aiplatform.googleapis.com.
	endpoint string
}

// Config groups Client construction inputs. Required + validated in New().
type Config struct {
	HTTPClient *http.Client  // optional; defaults to *http.Client w/ 30s timeout
	Tokens     TokenProvider // required
	Project    string        // required — GCP project (chora-489812)
	Location   string        // required — asia-southeast1 per ADR-163
	Endpoint   string        // optional — test-injection override
}

// New constructs a Client. Returns an error on missing required fields so
// the caller fails loud per feedback_no_stubs_real_wiring.
func New(cfg Config) (*Client, error) {
	if cfg.Tokens == nil {
		return nil, fmt.Errorf("gemini: TokenProvider required")
	}
	if cfg.Project == "" {
		return nil, fmt.Errorf("gemini: Project required")
	}
	if cfg.Location == "" {
		return nil, fmt.Errorf("gemini: Location required (e.g. asia-southeast1)")
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 30 * time.Second, Transport: newPooledTransport()}
	}
	return &Client{
		httpClient: hc,
		tokens:     cfg.Tokens,
		project:    cfg.Project,
		location:   cfg.Location,
		endpoint:   cfg.Endpoint,
	}, nil
}

// Family — implements domain.VendorClient.
func (c *Client) Family() domain.VendorFamily { return domain.VendorFamilyVertexGemini }

// newPooledTransport builds the default connection-pooled transport with a
// layered timeout hierarchy (connection / response-header / overall).
func newPooledTransport() *http.Transport {
	return &http.Transport{
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		ForceAttemptHTTP2:     true,
		DialContext: (&net.Dialer{
			Timeout: 10 * time.Second,
		}).DialContext,
	}
}

// Generate — implements domain.VendorClient.
func (c *Client) Generate(ctx context.Context, req domain.VendorRequest) (domain.VendorResponse, error) {
	if err := domain.CheckContextCancellation(ctx); err != nil {
		return domain.VendorResponse{}, err
	}
	model := string(req.LogicalModelID)

	host := c.endpoint
	if host == "" {
		// The `global` location is served by the un-prefixed global endpoint
		// (https://aiplatform.googleapis.com) — there is NO
		// `global-aiplatform.googleapis.com`. The URL path still carries
		// `locations/global`. Gemini 3.x text + image models are global-only
		// on chora-489812 (probed 2026-06-01); `global` is a verified strict
		// superset of asia-southeast1 + us-central1, so the gateway routes
		// every Vertex call through it. Regional locations keep the
		// `{location}-aiplatform.googleapis.com` host per ADR-163.
		if c.location == "global" {
			host = "https://aiplatform.googleapis.com"
		} else {
			host = fmt.Sprintf("https://%s-aiplatform.googleapis.com", c.location)
		}
	}
	url := fmt.Sprintf(
		"%s/v1/projects/%s/locations/%s/publishers/google/models/%s:generateContent",
		host, c.project, c.location, model,
	)

	body := buildGeminiBody(req)
	rawBody, err := json.Marshal(body)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("gemini: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(rawBody))
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("gemini: build request: %w", err)
	}
	token, err := c.tokens.Token(ctx)
	if err != nil {
		return domain.VendorResponse{}, fmt.Errorf("gemini: token: %w", err)
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
		return domain.VendorResponse{}, fmt.Errorf("gemini: HTTP do: %w", err)
	}
	defer func() { _ = httpResp.Body.Close() }()
	respBody, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode >= 400 {
		// Surface the vendor body so the service layer + fallback walk can
		// log the actual upstream error for diagnostic value. The status is
		// carried on the error so the adapter can relay it (401/403 →
		// Unauthenticated, 404 → NotFound, 429 → ResourceExhausted).
		return domain.VendorResponse{}, &domain.ProviderError{
			Status:   httpResp.StatusCode,
			Endpoint: httpReq.URL.String(),
			Body:     strings.TrimSpace(string(respBody)),
		}
	}
	var parsed geminiResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return domain.VendorResponse{}, fmt.Errorf("gemini: unmarshal response: %w", err)
	}
	return parsed.toDomain(model)
}

// ---------------------------------------------------------------------------
// Wire shapes — minimal subset of Vertex AI Gemini generateContent.
// ---------------------------------------------------------------------------

type geminiBody struct {
	Contents          []geminiContent `json:"contents"`
	SystemInstruction *geminiContent  `json:"systemInstruction,omitempty"`
	// Tools carries the genai []*Tool function declarations (ADR-177). Raw
	// passthrough — the genai-marshaled shape is already Vertex-native
	// (`[{functionDeclarations:[...]}]`).
	Tools            json.RawMessage `json:"tools,omitempty"`
	GenerationConfig map[string]any  `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text,omitempty"`
	// InlineData carries a base64-encoded inline blob — for image-modality
	// responses gemini returns the generated image here (W8, CR 2026-06-01).
	InlineData *geminiInlineData `json:"inlineData,omitempty"`
	// FunctionCall / FunctionResponse carry the tool-calling parts (ADR-177).
	// FunctionCall appears on `model` turns (the model asks to call a tool);
	// FunctionResponse on `user` turns (the agent returns the tool result).
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
	// FileData references an external blob by URI (Vertex AI's `fileData`
	// { mimeType, fileUri }). For EPIC-1a batch grounding the qgen agent injects
	// a gs:// fileData part so Vertex Gemini reads the uploaded source material
	// DIRECTLY from GCS — no inline base64 (which would blow the gRPC 4 MB
	// message cap on a large PDF). buildGeminiBody round-trips it verbatim
	// (unmarshal contents_json → re-marshal to the Vertex body).
	FileData *geminiFileData `json:"fileData,omitempty"`
}

type geminiFunctionCall struct {
	Name string         `json:"name"`
	Args map[string]any `json:"args,omitempty"`
}

type geminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response,omitempty"`
}

// geminiInlineData is the wire shape for an inline blob part — Vertex AI's
// `inlineData` { mimeType, data } where data is standard base64.
type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

// geminiFileData is the wire shape for a by-reference blob part — Vertex AI's
// `fileData` { mimeType, fileUri }. fileUri is a gs:// (or https) URI the model
// reads directly; the gateway never downloads it.
type geminiFileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content      geminiContent `json:"content"`
		FinishReason string        `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount        int64 `json:"promptTokenCount"`
		CandidatesTokenCount    int64 `json:"candidatesTokenCount"`
		CachedContentTokenCount int64 `json:"cachedContentTokenCount"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
}

func buildGeminiBody(req domain.VendorRequest) geminiBody {
	var body geminiBody
	// ADR-177 tool-calling path: when a multi-turn ContentsJSON is supplied build
	// the request from it (ignoring the flat Prompt). On a malformed payload fall
	// back to the legacy single-prompt path rather than dropping the turn.
	if strings.TrimSpace(req.ContentsJSON) != "" {
		var contents []geminiContent
		if err := json.Unmarshal([]byte(req.ContentsJSON), &contents); err == nil && len(contents) > 0 {
			body.Contents = contents
		}
	}
	if len(body.Contents) == 0 {
		body.Contents = []geminiContent{
			{Role: "user", Parts: []geminiPart{{Text: req.Prompt}}},
		}
	}
	// Tool declarations — raw passthrough of the genai []*Tool shape.
	if strings.TrimSpace(req.ToolsJSON) != "" {
		body.Tools = json.RawMessage(req.ToolsJSON)
	}
	if req.SystemPrompt != "" {
		body.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: req.SystemPrompt}},
		}
	}
	// generationConfig is the documented Vertex `:generateContent` location
	// for responseModalities (the model-level "what kinds of parts may the
	// candidate carry" selector). We inject it there for an IMAGE request.
	//
	// Clone the caller-supplied map before mutating: req.GenerationConfig is
	// the domain's shared map[string]any (one per Invoke, but the contract
	// makes no copy guarantee), so writing into it directly would be an
	// adapter mutating a domain value.
	if len(req.GenerationConfig) > 0 || isImageModality(req.ResponseModality) {
		genCfg := make(map[string]any, len(req.GenerationConfig)+1)
		for k, v := range req.GenerationConfig {
			genCfg[k] = v
		}
		if isImageModality(req.ResponseModality) {
			genCfg["responseModalities"] = []string{"IMAGE"}
		}
		body.GenerationConfig = genCfg
	}
	return body
}

// isImageModality reports whether the request asks for an IMAGE response.
// Case-insensitive on the modality string so a caller sending "image" or
// "IMAGE" both route to image generation.
func isImageModality(modality string) bool {
	return strings.EqualFold(modality, "IMAGE")
}

func (g geminiResponse) toDomain(fallbackModel string) (domain.VendorResponse, error) {
	resp := domain.VendorResponse{
		Usage: domain.TokenUsage{
			InputTokens:  g.UsageMetadata.PromptTokenCount,
			OutputTokens: g.UsageMetadata.CandidatesTokenCount,
			CachedTokens: g.UsageMetadata.CachedContentTokenCount,
			CostMicros:   geminiCostMicros(fallbackModel, g.UsageMetadata.PromptTokenCount, g.UsageMetadata.CandidatesTokenCount, g.UsageMetadata.CachedContentTokenCount),
		},
		ModelVersion: g.ModelVersion,
	}
	if resp.ModelVersion == "" {
		resp.ModelVersion = fallbackModel
	}
	if len(g.Candidates) > 0 {
		// A candidate may carry text parts, inlineData (image) parts, or both.
		// Accumulate text; for the FIRST inlineData part, base64-decode it into
		// the image carrier. A response may legitimately be image-only (no text).
		var sb strings.Builder
		var toolCalls []geminiFunctionCall
		for _, p := range g.Candidates[0].Content.Parts {
			sb.WriteString(p.Text)
			if p.FunctionCall != nil {
				toolCalls = append(toolCalls, *p.FunctionCall)
			}
			if p.InlineData != nil && resp.ImageBytes == nil {
				raw, err := base64.StdEncoding.DecodeString(p.InlineData.Data)
				if err != nil {
					// Fail loud per directive — a malformed inlineData blob is a
					// vendor-contract violation, not a silently-empty image.
					return domain.VendorResponse{}, fmt.Errorf("gemini: decode inlineData base64: %w", err)
				}
				resp.ImageBytes = raw
				resp.ImageMIMEType = p.InlineData.MimeType
			}
		}
		resp.Completion = sb.String()
		// ADR-177 — surface model-emitted tool calls so the agent can run them +
		// continue the loop. Marshaled in the genai functionCall shape the ADK
		// modelgatewayclient parses back into genai.Part{FunctionCall}.
		if len(toolCalls) > 0 {
			tc, err := json.Marshal(toolCalls)
			if err != nil {
				return domain.VendorResponse{}, fmt.Errorf("gemini: marshal tool calls: %w", err)
			}
			resp.ToolCallsJSON = string(tc)
		}
		resp.FinishReason = mapGeminiFinishReason(g.Candidates[0].FinishReason)
	} else {
		resp.FinishReason = domain.FinishReasonUnspecified
	}
	return resp, nil
}

func mapGeminiFinishReason(s string) domain.FinishReason {
	switch s {
	case "STOP":
		return domain.FinishReasonComplete
	case "MAX_TOKENS":
		return domain.FinishReasonMaxTokens
	case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT":
		return domain.FinishReasonModelArmorBlock
	default:
		return domain.FinishReasonUnspecified
	}
}

// geminiPrice is one model's rates in micro-USD per 1M tokens. Divide by
// 1_000_000 for the per-token cost.
type geminiPrice struct {
	InputMicrosPer1M  int64
	OutputMicrosPer1M int64
	// CacheMicrosPer1M prices context-cached input tokens. Cached input is
	// DISCOUNTED, not free — billing it at 0 (as this table did until
	// 2026-07-16) is an under-bill wearing a simplification's clothes.
	CacheMicrosPer1M int64
}

// geminiPricing maps a Vertex model ID to its billing rates. This table is the
// ONLY input to TokenUsageLedger cost_micros and to the per-tenant budget
// DebitSpent — a wrong or missing rate is a wrong invoice and a wrong spend cap,
// not a cosmetic defect.
//
// PROVENANCE (CHO-2220, 2026-07-16): rates come from the published Vertex AI
// price list (https://cloud.google.com/vertex-ai/generative-ai/pricing),
// cross-checked against the Billing Catalog API (service C7E2-9256-1C43
// "Vertex AI"). Not from memory. chora-489812 has no billing export, so no
// invoice-level reconciliation is possible — the page is the best authority.
//
// ⚠ WHERE THE TWO DISAGREE, THE PAGE WINS. The SKU catalogue keeps preview-era
// SKUs alongside GA ones, and its names are grammatically ambiguous:
//   - "Gemini 2.5 Flash GA Text Input" ($0.30) vs "Gemini 2.5 Flash Text Input"
//     ($0.15) — the un-suffixed one is pre-GA. gemini-2.5-flash is GA.
//   - "Gemini 2.5 Flash Image Input" ($0.15) parses as either (2.5 Flash Image)
//     (Input) or (2.5 Flash)(Image Input), and is a stale preview rate either
//     way; the page prices gemini-2.5-flash-image input at $0.30. Reading that
//     SKU as authoritative cost a 2x under-bill on this table's first pass.
//   - But "Gemini 3.0 Pro Image Input Caching" sits beside Text/Audio/Video
//     Input Caching — there the suffix is the input MODALITY, not the model.
//   Same words, different grammar. Read the whole SKU family, not one name.
//
// Rates were previously frozen at M15 Phase 2.2 (commit d1af0bb2c) and had
// drifted to under-bill EVERY model they covered — 2.5-flash by 4x on input and
// 8.3x on output while carrying 78% of live traffic. The "Phase 4 moves this to
// YAML" plan never landed: config/pricing.yaml exists but is wired only to the
// O+ agent-decision cost column, never to this path. Two tables, and until this
// story the accurate one was inert while the stale one billed.
//
//	model                   $/1M in   $/1M out   $/1M cached
//	gemini-2.5-flash           0.30      2.50        0.03
//	gemini-2.5-pro             1.25     10.00        0.13     (<=200K context tier)
//	gemini-2.5-flash-lite      0.10      0.40        0.01
//	gemini-3.5-flash           1.50      9.00        0.15     (Global; non-global +10%)
//	gemini-3.1-flash-lite      0.25      1.50        0.025    (Global)
//	gemini-2.5-flash-image     0.30     30.00        0.03     (image output; cache inferred)
//	gemini-3.1-flash-image     0.50     60.00        0.05     (image output)
//	gemini-3-pro-image         2.00    120.00        0.20     (image output)
//
// Image models bill a generated image as output tokens (~1290/image), so the
// per-1M output rate covers image generation directly. NB the page prices
// gemini-3-pro-image TEXT output at $12/1M — we carry the $120 image rate
// because the qgen scene path always sets response_modality=IMAGE. A caller
// that asks these models for text would be over-billed 10x; revisit if one ever
// does.
//
// Only gemini-2.5-flash-image's cache rate is inferred (from its 2.5-flash
// base — the page and catalogue give it none). It is never exercised: scene
// generation is single-turn, so CachedContentTokenCount is 0.
//
// Tiering is NOT modelled: gemini-2.5-pro doubles above a 200K context
// (in $2.50 / out $15.00) and 3.5-flash is +10% off-Global. Both would
// UNDER-bill. Our traffic is Global and well under 200K; if that changes the
// table needs a context/region dimension, not a fudged rate.
//
// DELIBERATELY ABSENT — do not "complete" this table with guesses:
//   - gemini-3.1-pro-preview: no SKU exists in the catalogue at all. $0 may be
//     correct while it is unbilled, and becomes wrong silently at GA — which is
//     what the unpriced warn below is for.
//   - gemini-3-flash-preview: a "Gemini 3 Flash" GA SKU family exists
//     ($0.50/$3.00) but a preview model cannot be confirmed to bill under it.
//     Guessing risks OVER-billing — fabrication in the other direction.
var geminiPricing = map[string]geminiPrice{
	"gemini-2.5-flash":       {300_000, 2_500_000, 30_000},
	"gemini-2.5-pro":         {1_250_000, 10_000_000, 130_000},
	"gemini-2.5-flash-lite":  {100_000, 400_000, 10_000},
	"gemini-3.5-flash":       {1_500_000, 9_000_000, 150_000},
	"gemini-3.1-flash-lite":  {250_000, 1_500_000, 25_000},
	"gemini-2.5-flash-image": {300_000, 30_000_000, 30_000},
	"gemini-3.1-flash-image": {500_000, 60_000_000, 50_000},
	"gemini-3-pro-image":     {2_000_000, 120_000_000, 200_000},
}

// resolveGeminiPriceKey maps a model id onto a key of geminiPricing, so a DATED
// SNAPSHOT bills as the base model it is a snapshot of (E7 shape (b)).
//
// WHY. geminiPricing is an exact map lookup. Without this, the moment anyone
// pins a versioned id (gemini-2.5-flash-002) the lookup misses and
// geminiCostMicros returns 0, so the caller bills $0 into the TokenUsageLedger
// AND debits $0 of tenant budget. Pinning a version to satisfy a governance
// requirement would silently turn a priced model into a free one, which is
// worse than the drift the pin exists to prevent. Vertex publishes no dated
// snapshot of these models TODAY, so this is deliberately landed ahead of the
// need: the trap springs on whoever pins first, and they should not have to
// know this file exists.
//
// WHAT IT REFUSES TO DO, because the table above forbids guessing a rate:
//   - It strips trailing NUMERIC segments only. "-preview" is never stripped,
//     so gemini-3.1-pro-preview and gemini-3-flash-preview stay unpriced, and a
//     dated preview (gemini-2.5-flash-preview-05-20) stops at its preview stem
//     rather than inheriting the GA base rate.
//   - It accepts a candidate ONLY if that candidate is already a curated key,
//     so it can never fabricate a rate for an unpriced family: gemini-4-flash-001
//     does not resolve.
//   - An exact entry always wins, so if a snapshot ever bills differently from
//     its base, adding that key overrides this entirely.
func resolveGeminiPriceKey(model string) (string, bool) {
	if _, ok := geminiPricing[model]; ok {
		return model, true
	}
	stem := model
	for {
		i := strings.LastIndex(stem, "-")
		if i <= 0 {
			return "", false
		}
		if !isAllDigits(stem[i+1:]) {
			return "", false
		}
		stem = stem[:i]
		if _, ok := geminiPricing[stem]; ok {
			return stem, true
		}
	}
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func geminiCostMicros(model string, inputTokens, outputTokens, cachedTokens int64) int64 {
	key, ok := resolveGeminiPriceKey(model)
	p := geminiPricing[key]
	if !ok {
		// Serve, but never silently: an unpriced model bills $0 into the
		// TokenUsageLedger AND debits $0 of tenant budget, so the spend is
		// invisible rather than free. Refusing here would turn an accounting
		// gap into an inference outage (a new preview model would break the
		// crew that calls it), so the gap is surfaced instead — owner decision,
		// CHO-2220. chora_unpriced_model is the stable field a Cloud Logging
		// log-based metric filters on; renaming it breaks the alert.
		slog.Warn("gemini: model has no price entry — billing $0 into the ledger and debiting $0 of budget",
			"model", model,
			"input_tokens", inputTokens,
			"output_tokens", outputTokens,
			"chora_unpriced_model", true,
		)
		return 0
	}
	// Cached input is billed ONCE, at the discounted cache rate — so the fresh
	// input is (total - cached), and the cached remainder is charged separately.
	// This table billed cached tokens at ZERO until 2026-07-16; Vertex publishes
	// real "... Input Caching" SKUs, so zero was an under-bill, not a
	// simplification.
	//
	// cachedTokens > inputTokens is nonsense the vendor should never send;
	// clamp to 0 rather than credit negative fresh input.
	effInput := inputTokens - cachedTokens
	if effInput < 0 {
		effInput = 0
	}
	in := (effInput * p.InputMicrosPer1M) / 1_000_000
	cached := (cachedTokens * p.CacheMicrosPer1M) / 1_000_000
	out := (outputTokens * p.OutputMicrosPer1M) / 1_000_000
	return in + cached + out
}
