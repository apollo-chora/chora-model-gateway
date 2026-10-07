// Package vertexembed implements domain.EmbeddingClient against the Vertex
// AI publisher-model `:predict` embeddings surface (G1'-1, owner-ruled
// 2026-08-07: embeddings ride the chokepoint; the direct calls this
// replaces lived in chora-consumption and chora-creation under the
// ADR-173 carve-out).
//
//	POST https://{location}-aiplatform.googleapis.com/v1/projects/{p}/locations/{l}/publishers/google/models/{model}:predict
//	body  = {"instances":[{"task_type":"...","content":"..."}],
//	         "parameters":{"outputDimensionality":768}}
//	resp  = {"predictions":[{"embeddings":{"statistics":{"token_count":N},
//	         "values":[...]}}]}
//
// Mirrors the gemini vendor adapter's transport shape (TokenProvider +
// project/location + endpoint override for tests).
package vertexembed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

// TokenProvider yields a Google OAuth2 bearer token (structural twin of the
// gemini adapter's interface; main.go passes the same concrete provider).
type TokenProvider interface {
	Token(ctx context.Context) (string, error)
}

// ProviderError mirrors the Python runtime's ProviderError: a non-2xx
// response from the provider. Carries the provider's own HTTP status so the
// adapter can relay it instead of flattening everything into one code.
type ProviderError struct {
	Status   int
	Endpoint string
	Body     string
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("upstream %d from %s: %s", e.Status, e.Endpoint, e.Body)
}

// UpstreamStatus exposes the provider's HTTP status for status mapping.
func (e *ProviderError) UpstreamStatus() int { return e.Status }

// Config groups construction inputs; required fields validated in New().
type Config struct {
	HTTPClient *http.Client  // optional; defaults to 30s-timeout client
	Tokens     TokenProvider // required
	Project    string        // required
	Location   string        // required (asia-southeast1 per ADR-163)
	Endpoint   string        // optional test override, e.g. httptest URL
}

// Client is the Vertex embeddings EmbeddingClient.
type Client struct {
	httpClient *http.Client
	tokens     TokenProvider
	project    string
	location   string
	endpoint   string
}

// New constructs the client, failing loud on missing required config.
func New(cfg Config) (*Client, error) {
	if cfg.Tokens == nil {
		return nil, fmt.Errorf("vertexembed: TokenProvider required")
	}
	if cfg.Project == "" {
		return nil, fmt.Errorf("vertexembed: Project required")
	}
	if cfg.Location == "" {
		return nil, fmt.Errorf("vertexembed: Location required")
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

// Family — implements domain.EmbeddingClient.
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

type predictInstance struct {
	TaskType string `json:"task_type,omitempty"`
	Content  string `json:"content"`
}

type predictParameters struct {
	OutputDimensionality int32 `json:"outputDimensionality"`
}

type predictRequest struct {
	Instances  []predictInstance `json:"instances"`
	Parameters predictParameters `json:"parameters"`
}

type predictResponse struct {
	Predictions []struct {
		Embeddings struct {
			Statistics struct {
				TokenCount int64 `json:"token_count"`
			} `json:"statistics"`
			Values []float32 `json:"values"`
		} `json:"embeddings"`
	} `json:"predictions"`
}

// EmbedText — implements domain.EmbeddingClient.
func (c *Client) EmbedText(ctx context.Context, req domain.EmbedVendorRequest) (domain.EmbedVendorResponse, error) {
	if err := domain.CheckContextCancellation(ctx); err != nil {
		return domain.EmbedVendorResponse{}, err
	}
	taskType := req.TaskType
	if taskType == "" {
		taskType = "RETRIEVAL_DOCUMENT"
	}
	body, err := json.Marshal(predictRequest{
		Instances:  []predictInstance{{TaskType: taskType, Content: req.Text}},
		Parameters: predictParameters{OutputDimensionality: req.OutputDimensions},
	})
	if err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("vertexembed: marshal: %w", err)
	}

	base := c.endpoint
	if base == "" {
		base = fmt.Sprintf("https://%s-aiplatform.googleapis.com", c.location)
	}
	url := fmt.Sprintf("%s/v1/projects/%s/locations/%s/publishers/google/models/%s:predict",
		base, c.project, c.location, string(req.LogicalModelID))

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("vertexembed: build request: %w", err)
	}
	tok, err := c.tokens.Token(ctx)
	if err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("vertexembed: token: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+tok)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("vertexembed: predict call: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return domain.EmbedVendorResponse{}, &ProviderError{
			Status:   resp.StatusCode,
			Endpoint: httpReq.URL.String(),
			Body:     strings.TrimSpace(string(snippet)),
		}
	}

	var out predictResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return domain.EmbedVendorResponse{}, fmt.Errorf("vertexembed: decode: %w", err)
	}
	if len(out.Predictions) == 0 || len(out.Predictions[0].Embeddings.Values) == 0 {
		return domain.EmbedVendorResponse{}, fmt.Errorf("vertexembed: empty embedding prediction")
	}
	return domain.EmbedVendorResponse{
		Values:       out.Predictions[0].Embeddings.Values,
		ModelVersion: string(req.LogicalModelID),
		InputTokens:  out.Predictions[0].Embeddings.Statistics.TokenCount,
	}, nil
}

// Compile-time port guarantee.
var _ domain.EmbeddingClient = (*Client)(nil)
