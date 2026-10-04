// Package main is the chora-model-gateway entrypoint.
//
// It wires the adapters into domain.Service and serves two surfaces:
//
//	gRPC  :9090   ModelGatewayService.Invoke + .Embed
//	HTTP  :8080   the OpenAI-compatible /v1 API + /healthz + /readyz
//
// There is no cloud dependency. The gateway's infrastructure is:
//
//	Postgres        per-tenant budget + the token-usage cost ledger
//	the environment  credentials (a registry entry names an env var)
//	the registry     which model maps to which provider, endpoint and limits
//
// Everything else — which provider, which endpoint, which model name, the
// context window, the output ceiling, the per-token price — is a config
// value, never a compiled-in default. See config/models.yaml.
package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"

	mgv1 "github.com/locoroco-git/Chora-LMS/chora-contracts/gen/go/chora/services/model_gateway/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/grpc"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/httpapi"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/pg"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/policy"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/secrets"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/anthropic"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/openai"
	"github.com/apollo-chora/chora-model-gateway/internal/config"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
)

func main() {
	// `-healthcheck` makes the binary its own Docker HEALTHCHECK.
	//
	// The runtime image is distroless: no shell, no curl, no wget for a
	// healthcheck to call. Giving the binary the flag means the image is
	// health-checkable without adding a single package to it.
	if len(os.Args) == 2 && os.Args[1] == "-healthcheck" {
		os.Exit(runHealthcheck())
	}

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	if err := run(); err != nil {
		logger.Error("gateway exited with error", "err", err)
		os.Exit(1)
	}
}

// runHealthcheck probes the local /healthz endpoint and returns a process exit
// code.
func runHealthcheck() int {
	addr := net.JoinHostPort("127.0.0.1", envOr("CHORA_HTTP_PORT", "8080"))

	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get("http://" + addr + "/healthz") // #nosec G107 -- loopback, fixed path
	if err != nil {
		fmt.Fprintf(os.Stderr, "healthcheck: %v\n", err)
		return 1
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "healthcheck: status %d\n", resp.StatusCode)
		return 1
	}
	return 0
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// ---------------------------------------------------------------------
	// Config + registry. Both fail loud and early: a bad registry is an
	// operator mistake that should surface before the first request, not as
	// a 502 twenty minutes later.
	// ---------------------------------------------------------------------
	cfg, err := config.LoadConfig()
	if err != nil {
		return err
	}
	registry, err := config.LoadRegistry(cfg.RegistryPath)
	if err != nil {
		return err
	}
	slog.Info("gateway boot",
		"version", cfg.ServiceVersion,
		"registry", cfg.RegistryPath,
		"models", len(registry.Names()),
		"grpc_port", cfg.GRPCPort,
		"http_port", cfg.HTTPPort,
		"default_tenant", cfg.DefaultTenantID,
	)

	// Report the credential situation per model WITHOUT printing a secret, so
	// a misconfigured key is visible at boot rather than on the first 401.
	credentialResolver := secrets.NewEnvResolver()
	for _, target := range registry.Targets() {
		slog.Info("model registered",
			"model", target.LogicalModelID,
			"provider", target.Vendor,
			"upstream_model", target.UpstreamModel,
			"endpoint", primaryEndpointFor(target),
			"capabilities", target.Capabilities,
			"context_window", target.ContextWindow,
			"max_output_tokens", target.MaxOutputTokens,
			"credential", credentialResolver.Describe(target.APIKeyRef),
		)
	}

	// ---------------------------------------------------------------------
	// Database. Fail-loud: a gateway that cannot record what it spent is
	// worse than one that refuses to start, because the caller would get
	// completions that nobody is billed for.
	// ---------------------------------------------------------------------
	repo, closeDB, err := openRepo(cfg.DatabaseURL, cfg.BootstrapTimeout)
	if err != nil {
		return err
	}
	defer closeDB()

	// ---------------------------------------------------------------------
	// Policy loader + vendor adapters.
	// ---------------------------------------------------------------------
	loader, err := policy.NewLoader(registry)
	if err != nil {
		return fmt.Errorf("policy loader: %w", err)
	}

	vendorHTTPClient := &http.Client{
		// No client-level timeout: the per-request context carries the
		// deadline. Setting one here would cap a legitimately slow local
		// model server as aggressively as a fast hosted one.
		Transport: defaultTransport(),
	}

	openAIClient, err := openai.New(openai.Config{
		HTTPClient: vendorHTTPClient,
		Pricing:    registry,
	})
	if err != nil {
		return fmt.Errorf("openai adapter: %w", err)
	}
	anthropicClient, err := anthropic.New(anthropic.Config{
		HTTPClient: vendorHTTPClient,
		Pricing:    registry,
	})
	if err != nil {
		return fmt.Errorf("anthropic adapter: %w", err)
	}

	// ---------------------------------------------------------------------
	// Domain service.
	// ---------------------------------------------------------------------
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{openAIClient, anthropicClient},
		Budget:         repo,
		Outbox:         repo,
		Secrets:        credentialResolver,
		Policies:       loader,
		Claims:         repo,
		Embedder:       openAIClient,
		GatewayVersion: cfg.ServiceVersion,
		Now:            time.Now,
		NewID:          newInvocationID,
	})
	if err != nil {
		return fmt.Errorf("domain service: %w", err)
	}

	// ---------------------------------------------------------------------
	// gRPC surface.
	// ---------------------------------------------------------------------
	gwAdapter, err := modelgatewaygrpc.NewServer(svc, svc)
	if err != nil {
		return fmt.Errorf("grpc adapter: %w", err)
	}
	gs := grpc.NewServer(
		// TLS is the deploy target's business (a mesh sidecar, a reverse
		// proxy, or nothing at all when bound to localhost). Binding a
		// self-signed cert here would move that decision into the binary.
		grpc.Creds(insecure.NewCredentials()),
	)
	mgv1.RegisterModelGatewayServiceServer(gs, gwAdapter)

	hsrv := health.NewServer()
	healthpb.RegisterHealthServer(gs, hsrv)
	hsrv.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	hsrv.SetServingStatus("chora.services.model_gateway.v1.ModelGatewayService", healthpb.HealthCheckResponse_SERVING)

	grpcListener, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.GRPCPort))
	if err != nil {
		return fmt.Errorf("grpc listen :%d: %w", cfg.GRPCPort, err)
	}

	// ---------------------------------------------------------------------
	// OpenAI-compatible HTTP surface.
	// ---------------------------------------------------------------------
	facade, err := httpapi.NewServer(httpapi.Config{
		Gateway:         svc,
		Models:          registry,
		Version:         cfg.ServiceVersion,
		DefaultTenantID: cfg.DefaultTenantID,
		DefaultGCID:     cfg.DefaultGCID,
		DefaultAgentID:  cfg.DefaultAgentID,
	})
	if err != nil {
		return fmt.Errorf("http facade: %w", err)
	}
	httpServer := &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.HTTPPort),
		Handler:           facade,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       5 * time.Minute,
		WriteTimeout:      5 * time.Minute,
		IdleTimeout:       2 * time.Minute,
	}

	// ---------------------------------------------------------------------
	// Serve both, then wait for a signal or a fatal serve error.
	// ---------------------------------------------------------------------
	errCh := make(chan error, 2)
	go func() {
		slog.Info("gRPC listening", "addr", grpcListener.Addr().String())
		errCh <- gs.Serve(grpcListener)
	}()
	go func() {
		slog.Info("OpenAI-compatible API listening", "addr", httpServer.Addr, "base_url", fmt.Sprintf("http://localhost:%d/v1", cfg.HTTPPort))
		err := httpServer.ListenAndServe()
		if errors.Is(err, http.ErrServerClosed) {
			err = nil
		}
		errCh <- err
	}()

	select {
	case <-ctx.Done():
		slog.Info("shutdown signal received")
	case err := <-errCh:
		if err != nil {
			return err
		}
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	stopped := make(chan struct{})
	go func() {
		gs.GracefulStop()
		close(stopped)
	}()
	select {
	case <-stopped:
	case <-shutdownCtx.Done():
		gs.Stop()
	}
	if err := httpServer.Shutdown(shutdownCtx); err != nil {
		slog.Warn("http shutdown", "err", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Boot helpers
// ---------------------------------------------------------------------------

// openRepo connects to Postgres and waits for it to answer.
//
// The retry loop matters in a container start: Postgres and the gateway come
// up together, and on a cold machine the database is routinely not accepting
// connections for several seconds. Failing on the first refused connection
// would turn a normal cold start into a restart loop.
func openRepo(dsn string, timeout time.Duration) (*pg.Repo, func(), error) {
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return nil, nil, fmt.Errorf("sql.Open: %w", err)
	}

	pingCtx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	var lastErr error
	for {
		// Each attempt gets its own deadline so one hung dial cannot swallow
		// the whole budget.
		attemptCtx, attemptCancel := context.WithTimeout(pingCtx, 5*time.Second)
		err := db.PingContext(attemptCtx)
		attemptCancel()
		if err == nil {
			break
		}
		lastErr = err
		select {
		case <-pingCtx.Done():
			_ = db.Close()
			return nil, nil, fmt.Errorf("database unreachable after %s: %w", timeout, lastErr)
		case <-time.After(250 * time.Millisecond):
		}
	}

	repo, err := pg.New(db)
	if err != nil {
		_ = db.Close()
		return nil, nil, err
	}
	slog.Info("database connected")
	return repo, func() { _ = db.Close() }, nil
}

// primaryEndpointFor names the endpoint a model's PRIMARY capability will
// actually hit.
//
// It is deliberately not just "the chat URL": an image-only entry posts to
// /images/generations and an embedding-only entry to /embeddings, so logging
// the text endpoint for those sends an operator debugging in exactly the wrong
// direction.
func primaryEndpointFor(t domain.TargetModel) string {
	capabilities := t.Capabilities
	if len(capabilities) == 0 {
		capabilities = []string{domain.CapabilityChat}
	}
	for _, c := range capabilities {
		switch c {
		case domain.CapabilityImage:
			return t.ImagesURL()
		case domain.CapabilityEmbeddings:
			if !t.Supports(domain.CapabilityChat) {
				return t.EmbeddingsURL()
			}
		case domain.CapabilityWebSearch:
			if t.Grounding != nil {
				return t.GroundingURL()
			}
		}
	}
	return t.PrimaryDispatchURL()
}
func newInvocationID() string {
	id, err := uuid.NewV7()
	if err != nil {
		// crypto/rand exhaustion is not recoverable here, but a v4 still gives
		// the ledger a unique key, which is the property that matters.
		return uuid.NewString()
	}
	return id.String()
}

func defaultTransport() http.RoundTripper {
	// A tuned-but-default transport: raise the idle-connection pool so a
	// burst of parallel calls does not open a fresh TLS handshake per
	// request, and bound the idle window so a rotated endpoint's dead
	// connections are reaped.
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.MaxIdleConns = 100
	transport.MaxIdleConnsPerHost = 32
	transport.IdleConnTimeout = 90 * time.Second
	return transport
}

// envOr reads an environment variable, falling back when unset or blank.
func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}
