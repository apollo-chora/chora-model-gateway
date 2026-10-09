// Package main is the chora-model-gateway service entrypoint.
//
// Wires all adapters into domain.Service + serves ModelGatewayService.Invoke
// on gRPC :9090 (port name "grpc" appProtocol grpc per ADR-140) + plain HTTP
// /healthz (liveness) and /readyz (DB-backed readiness) on :8080 for kubelet
// probes.
//
// No inline config per CLAUDE.md §6 + secrets-and-env skill — every
// runtime value (project, region, DSN secret id) comes from env vars
// sourced from Terraform (non-secret) or Secret Manager (secrets).
//
// Phase 2.4 wires the production shape. Phase 2.5 builds the linux/amd64
// image + applies the Phase 0.5 manifests + grpcurls the LIVE pod.
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
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
	_ "github.com/jackc/pgx/v5/stdlib"
	"go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"

	commonobs "github.com/apollo-chora/chora-common/observability"
	mgv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"

	"github.com/apollo-chora/chora-model-gateway/internal/adapter/clients"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/grpc"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/middleware"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/modelarmor"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/pg"
	registryadapter "github.com/apollo-chora/chora-model-gateway/internal/adapter/registry"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/secrets"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/anthropic"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/gemini"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/openai"
	"github.com/apollo-chora/chora-model-gateway/internal/adapter/vendors/vertexembed"
	openaiapi "github.com/apollo-chora/chora-model-gateway/internal/api/openai"
	"github.com/apollo-chora/chora-model-gateway/internal/domain"
	"github.com/apollo-chora/chora-model-gateway/internal/executor"
	"github.com/apollo-chora/chora-model-gateway/internal/registry"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)

	if err := run(); err != nil {
		logger.Error("gateway shutdown with error", "err", err)
		os.Exit(1)
	}
}

func run() error {
	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	cfg, err := loadConfig()
	if err != nil {
		return fmt.Errorf("config: %w", err)
	}
	slog.Info("gateway boot",
		"project", cfg.project,
		"vertex_ai_location", cfg.vertexAILocation,
		"embed_vertex_location", cfg.embedVertexLocation,
		"model_armor_location", cfg.modelArmorLocation,
		"environment", cfg.environment,
		"gateway_version", cfg.gatewayVersion,
		"budget_required", cfg.budgetRequired,
	)

	// ---------------------------------------------------------------------
	// OTLP wiring — direct to Cloud Trace per Tier 3 D13 + Wave B fix
	// (2026-05-14). Uses the canonical chora-common/observability
	// InitOTLPAsync helper which:
	//   - Routes to the Cloud Trace exporter (NOT raw otlptracegrpc — that
	//     silently dropped spans pre-Wave-B per the package docs).
	//   - Falls back to stdouttrace when GOOGLE_CLOUD_PROJECT + ADC are unset.
	//   - Registers W3C TraceContext + Baggage propagators so inbound
	//     traceparent headers are honoured.
	// Fail-soft: OTLP init runs in its own goroutine + degrades to no-op
	// on timeout. The rest of bootstrap (token source, secrets, adapters)
	// keeps the full deadline budget.
	// ---------------------------------------------------------------------
	otlpHandle := commonobs.InitOTLPAsync(ctx, "chora-model-gateway", cfg.gatewayVersion)
	defer func() {
		shutdownCtx, c := context.WithTimeout(context.Background(), 5*time.Second)
		defer c()
		res := otlpHandle.WaitContext(shutdownCtx)
		if err := res.Shutdown(shutdownCtx); err != nil {
			slog.Warn("trace shutdown error", "err", err)
		}
	}()

	// ---------------------------------------------------------------------
	// Token source — Workload Identity Federation via google.DefaultTokenSource.
	// At runtime in GKE Autopilot the KSA→GSA binding makes ADC return a
	// chora-model-gateway-scoped TokenSource. On a dev laptop with
	// GOOGLE_APPLICATION_CREDENTIALS set, ADC returns the dale-cli source.
	// ---------------------------------------------------------------------
	tokens, err := newTokenProvider(ctx)
	if err != nil {
		return fmt.Errorf("token provider: %w", err)
	}

	// ---------------------------------------------------------------------
	// Secret Manager client (BYOA + global key resolution).
	// ---------------------------------------------------------------------
	secretClient, err := secrets.New(secrets.Config{
		Tokens:      tokens,
		Project:     cfg.project,
		Environment: cfg.environment,
	})
	if err != nil {
		return fmt.Errorf("secrets adapter: %w", err)
	}

	// ---------------------------------------------------------------------
	// Cloud Model Armor adapter — pinned to the regional endpoint per
	// feedback_model_armor_regional_endpoint.
	// ---------------------------------------------------------------------
	armorClient, err := modelarmor.New(modelarmor.Config{
		Tokens:   tokens,
		Location: cfg.modelArmorLocation,
	})
	if err != nil {
		return fmt.Errorf("modelarmor adapter: %w", err)
	}

	// ---------------------------------------------------------------------
	// Vendor adapters (3).
	// ---------------------------------------------------------------------
	geminiClient, err := gemini.New(gemini.Config{
		HTTPClient: &http.Client{Timeout: cfg.vendorHTTPTimeout},
		Tokens:     tokens,
		Project:    cfg.project,
		Location:   cfg.vertexAILocation,
	})
	if err != nil {
		return fmt.Errorf("gemini adapter: %w", err)
	}
	openaiClient, err := openai.New(openai.Config{Secrets: secretClient})
	if err != nil {
		return fmt.Errorf("openai adapter: %w", err)
	}
	anthropicClient, err := anthropic.New(anthropic.Config{Secrets: secretClient})
	if err != nil {
		return fmt.Errorf("anthropic adapter: %w", err)
	}

	// G1'-1 (2026-08-07): the embeddings chokepoint re-route. Same project,
	// location and token provider as the gemini adapter; the domain Embed
	// flow enters at permissive (no Armor leg, Bypassed markers) and
	// attaches no price, but every call produces a ledger row.
	embedClient, err := vertexembed.New(vertexembed.Config{
		HTTPClient: &http.Client{Timeout: cfg.vendorHTTPTimeout},
		Tokens:     tokens,
		Project:    cfg.project,
		Location:   cfg.embedVertexLocation,
	})
	if err != nil {
		return fmt.Errorf("vertexembed adapter: %w", err)
	}

	// ---------------------------------------------------------------------
	// Postgres adapter — opens *sql.DB to chora_observability via the
	// cloudsql-proxy sidecar at 127.0.0.1:5432. DSN is FAIL-LOUD: fetched
	// from Secret Manager by the secret ID in CHORA_OBSERVABILITY_DSN_SECRET_ID
	// at boot. Per feedback_no_stubs_real_wiring no fallback to in-memory or
	// no-op repos — if the DSN is unset, unfetchable, or the DB ping fails,
	// the service exits non-zero so kubelet surfaces the misconfiguration as
	// CrashLoopBackOff (the operational signal the user can see).
	// ---------------------------------------------------------------------
	// Retry loop on the DSN fetch — mirrors the openRepo() retry pattern.
	// The token-mint path inside ResolveSecret dials the GKE metadata-server
	// at 169.254.169.254:80, which momentarily refuses connections on cold-
	// start nodes (Pod scheduled before metadata-server daemonset's iptables
	// rules take effect). The default behavior is fail-fast on conn-refused
	// → service exits → 1 CrashLoop restart → 2nd attempt succeeds. This
	// loop bounds the retry to cfg.dbConnectTimeout (90s default) and
	// retries every 500ms, so the FIRST attempt succeeds gracefully without
	// a CrashLoop. Backward-compatible — once the metadata-server is up the
	// first ResolveSecret succeeds and the loop exits immediately.
	dsnFetchCtx, dsnFetchCancel := context.WithTimeout(ctx, cfg.dbConnectTimeout)
	defer dsnFetchCancel()
	var (
		dsn     string
		lastErr error
	)
	for {
		attemptCtx, attemptCancel := context.WithTimeout(dsnFetchCtx, 10*time.Second)
		dsn, lastErr = secretClient.ResolveSecret(attemptCtx, cfg.observabilityDSNSecretID)
		attemptCancel()
		if lastErr == nil {
			break
		}
		select {
		case <-dsnFetchCtx.Done():
			return fmt.Errorf("postgres adapter: fetch DSN secret %q (gave up after %s): %w",
				cfg.observabilityDSNSecretID, cfg.dbConnectTimeout, lastErr)
		case <-time.After(500 * time.Millisecond):
			// retry — likely transient metadata-server cold-start
		}
	}
	repo, dbCleanup, err := openRepo(dsn, cfg.dbConnectTimeout)
	if err != nil {
		return fmt.Errorf("postgres adapter: %w", err)
	}
	defer dbCleanup()

	// ---------------------------------------------------------------------
	// Model registry — the YAML-based catalogue the domain service resolves
	// model metadata (capabilities, output ceiling, credentials, grounding)
	// through. Loaded from CHORA_MODEL_REGISTRY (default config/models.yaml).
	// ---------------------------------------------------------------------
	modelRegistry, err := registry.Load(envOr("CHORA_MODEL_REGISTRY", "config/models.yaml"))
	if err != nil {
		return fmt.Errorf("model registry: %w", err)
	}
	modelResolver, err := registryadapter.NewResolver(modelRegistry)
	if err != nil {
		return fmt.Errorf("model registry adapter: %w", err)
	}

	// ---------------------------------------------------------------------
	// Policy loader — Phase 2.4 ships an in-memory default per agent.
	// Phase 4 follow-up moves this to a GCS-managed YAML per the
	// agent-guardrail-mapping.yaml convention (ADR-152 §"Mapping").
	// ---------------------------------------------------------------------
	policies := newDefaultPolicyLoader(cfg)

	// ---------------------------------------------------------------------
	// WS-1 umbrella metering client (identity ManaService). Constructed BEFORE
	// the domain service so its domain-typed view (DomainManaMeter) can back the
	// GroundedSearch fail-closed mana gate (ADR-231 / ADR-177 central metering);
	// the ManaMetering decorator (below) wraps the SAME client for Invoke.
	// ---------------------------------------------------------------------
	manaClient, manaConnClose, err := clients.NewManaClientFromAddr(cfg.identityGRPCAddr)
	if err != nil {
		return fmt.Errorf("mana metering client: %w", err)
	}
	defer func() {
		if cerr := manaConnClose(); cerr != nil {
			slog.Warn("model-gateway: mana client conn.Close", "err", cerr)
		}
	}()

	// ---------------------------------------------------------------------
	// Domain service composition. The single *pg.Repo backs Budget + Outbox
	// (Invoke) AND ExternalEgressGate + EgressAuditWriter (GroundedSearch,
	// ADR-231); geminiClient backs BOTH the Vendor list AND the grounded vendor
	// port (google_search grounding); DomainManaMeter is the grounded chain's
	// mana port.
	// ---------------------------------------------------------------------
	svc, err := domain.NewService(domain.ServiceConfig{
		Vendors:        []domain.VendorClient{geminiClient, openaiClient, anthropicClient},
		Armor:          armorClient,
		Budget:         repo,
		Outbox:         repo,
		Policies:       policies,
		GatewayVersion: cfg.gatewayVersion,
		EgressGate:     repo,
		Grounded:       geminiClient,
		Mana:           clients.NewDomainManaMeter(manaClient),
		EgressAudit:    repo,
		// ADR-152 amendment 2026-08-07 (G2): the gateway is the producer of
		// record for a Model Armor BLOCK. Same *pg.Repo, same outbox_events
		// table; the dispatcher publishes by row.Topic.
		Violations: repo,
		Embedder:   embedClient,
		// R22 (ADR-254 D7): the domain service takes the keyed debit claim
		// (gcid, dispatch_idempotency_key, action_code) BEFORE any spend and
		// reports a redelivery on InvokeResponse.Deduped — one claim
		// suppresses BOTH the tenant-budget debit and the mana debit.
		Claims: repo,
		// Transactional outbox: the budget debit + token-usage outbox row
		// land in ONE database transaction, with the debit gated on the
		// insert (a redelivery cannot debit twice).
		AtomicOutbox: repo,
		// G1'-3: promote the derived-contents Armor leg from audit-only to
		// blocking. Default false per the owner's closing rule of 2026-08-07
		// (never tighten an existing rule; a newly screened path must not start
		// refusing calls that pass today). The verdict is computed, returned
		// and published as governance evidence either way, so promotion can be
		// decided on measured data.
		EnforceContentsScreen: cfg.enforceContentsScreen,
		// G1'-2: promote the tool-call Armor leg from audit-only to blocking,
		// on the same terms and for the same reason as the line above.
		EnforceToolCallScreen: cfg.enforceToolCallScreen,
		EnforceToolsScreen:    cfg.enforceToolsScreen,
		// Cross-provider fallback: the FallbackEngine's same-provider retry
		// budget. 0 (the default) means a retryable failure advances straight
		// to the next target in the chain.
		FallbackRetries:      cfg.fallbackRetries,
		FallbackRetryBackoff: cfg.fallbackRetryBackoff,
		// Fail-closed "budget required" mode: a missing active budget window
		// is a BLOCK, not an allow — the absence of a policy must not restore
		// unlimited provider spending. Default OFF.
		BudgetRequired: cfg.budgetRequired,
		Now:                  time.Now,
		NewID:                newInvocationID,
		// The model-registry resolver backs the capability / output-ceiling /
		// credential checks + the budget-downgrade re-resolution.
		Models: modelResolver,
		// IMAGE action_code defaulting: "{defaultAgentID}_image".
		DefaultAgentID: envOr("CHORA_DEFAULT_AGENT_ID", "openai_compat"),
	})
	if err != nil {
		return fmt.Errorf("domain service: %w", err)
	}

	// ---------------------------------------------------------------------
	// Executor — the gateway's core execution engine. It composes the
	// governance pipeline (CompanionSuspension → ManaMetering → domain.Service)
	// into a single Execute() path. Both the gRPC and HTTP adapters call
	// Executor.Execute() — the SAME path — so Armor, mana, suspension and
	// budget gates apply identically regardless of transport.
	//
	// WS-1 umbrella metering (ADR-142 §4): the INVOKE path is decorated with
	// per-GCID mana debit. GroundedSearch self-meters its own high-price
	// external_egress action inside the domain chain, so it is NOT wrapped
	// here (wrapping would double-meter). The decorator pre-flights fail-closed
	// (premium learner) actions against the identity ManaService dry-run gate +
	// debits post-success; authoring/instructor actions are fail-open.
	//
	// ADR-252 D1 via ADR-254 D7: the Learning Companion containment read is a
	// decorator AHEAD of ManaMetering (deny-before-debit). It reads the two
	// chora_observability tables (migration 0018) uncached on every companion
	// turn through the same *pg.Repo, refuses an ABSENT surface loudly
	// (surface_unstamped) and a contained turn honestly (companion_suspended),
	// and fails CLOSED on a read error. The gateway is the control; the
	// operator write path lives in chora-observability.
	// ---------------------------------------------------------------------
	meteringCfg := middleware.DefaultConfig()
	meteringCfg.StrictOnError = cfg.manaStrictOnError
	meteringCfg.NewID = newInvocationID
	// ADR-254 D7 (R22): the keyed debit claim lives on the same *pg.Repo
	// (chora_observability migration 0019), so a redelivered dispatch bills once.
	meteringCfg.Claims = repo
	exec := executor.NewExecutor(svc, manaClient, meteringCfg, repo, middleware.SuspensionConfig{})
	slog.Info("model-gateway: Executor wired (suspension → mana → domain)",
		"identity_grpc_addr", cfg.identityGRPCAddr, "strict_on_error", cfg.manaStrictOnError,
		"fail_closed_prefix", meteringCfg.FailClosedPrefix, "debit_claims", "pg")

	// ---------------------------------------------------------------------
	// gRPC server. Invoke rides the Executor (suspension + mana decorators);
	// GroundedSearch calls the raw domain service (self-metering) directly
	// (ADR-231); Embed calls the raw domain service too (self-ledgering,
	// unpriced, G1'-1).
	// ---------------------------------------------------------------------
	gwAdapter, err := modelgatewaygrpc.NewServer(exec, svc, svc)
	if err != nil {
		return fmt.Errorf("grpc adapter: %w", err)
	}
	gs := grpc.NewServer(
		grpc.Creds(insecure.NewCredentials()), // mTLS terminates at Istio sidecar
		// otelgrpc server handler: starts a real OTel server span per RPC and
		// extracts inbound W3C TraceContext into the OTel context (a child of
		// the caller's trace when propagated). This is what gives Invoke a
		// valid, caller-correlated span — the source resolveTraceparent reads
		// to stamp every emitted token_usage event with a non-empty traceparent
		// (the shared outbox publisher rejects empty traceparent; see
		// HANDOFF_OBSERVABILITY_OUTBOX_JAM_2026-05-29 Fix 2).
		grpc.StatsHandler(otelgrpc.NewServerHandler()),
		// W3C traceparent string extraction onto ctx for slog trace correlation
		// per the canonical chora-common/observability convention.
		grpc.UnaryInterceptor(commonobs.GRPCServerInterceptor()),
	)
	mgv1.RegisterModelGatewayServiceServer(gs, gwAdapter)
	hsrv := health.NewServer()
	healthpb.RegisterHealthServer(gs, hsrv)
	hsrv.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	hsrv.SetServingStatus("chora.services.model_gateway.v1.ModelGatewayService", healthpb.HealthCheckResponse_SERVING)

	grpcListener, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.grpcPort))
	if err != nil {
		return fmt.Errorf("grpc listen: %w", err)
	}

	// ---------------------------------------------------------------------
	// OpenAI-compatible HTTP API. The server rides the SAME Executor as gRPC:
	// the invoker is the Executor (CompanionSuspension → ManaMetering →
	// domain.Service), the embedder is the raw domain service (Embed,
	// self-ledgering, unpriced — the same wiring the gRPC Embed path uses).
	// All business logic + governance lives in the Executor; the HTTP adapter
	// is a thin translation shim. Both gRPC and HTTP go through the exact
	// same Execute() path.
	// ---------------------------------------------------------------------
	openaiServer, err := openaiapi.NewServer(exec, svc, modelRegistry, repo, openaiapi.Settings{
		DefaultTenantID: cfg.defaultTenantID,
		DefaultGCID:     cfg.defaultGCID,
		DefaultAgentID:  cfg.defaultAgentID,
	})
	if err != nil {
		return fmt.Errorf("openai http server: %w", err)
	}

	httpServer := &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.httpPort),
		Handler:           openaiServer.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
	}

	errCh := make(chan error, 2)
	go func() {
		slog.Info("gRPC listening", "addr", grpcListener.Addr().String())
		errCh <- gs.Serve(grpcListener)
	}()
	go func() {
		slog.Info("HTTP /healthz + /readyz listening", "addr", httpServer.Addr)
		err := httpServer.ListenAndServe()
		if errors.Is(err, http.ErrServerClosed) {
			err = nil
		}
		errCh <- err
	}()

	select {
	case <-ctx.Done():
		slog.Info("graceful shutdown started")
	case err := <-errCh:
		if err != nil {
			return err
		}
	}

	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()
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
	_ = httpServer.Shutdown(shutdownCtx)
	return nil
}

// handleReadyz is the DB-backed readiness probe (Python parity:
// app/http.py ready() → database.ping()). Readiness = database reachable and
// usable: the gateway requires the DB for budget enforcement, debit
// idempotency and the usage outbox, so a model call that succeeds while
// settlement cannot be recorded is a billing-correctness failure — the
// gateway must not report ready then.
func handleReadyz(repo *pg.Repo) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := repo.Ping(r.Context()); err != nil {
			slog.Warn("readyz: database not reachable", "err", err)
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = fmt.Fprint(w, "not ready")
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = fmt.Fprint(w, "ready")
	}
}

// ----------------------------------------------------------------------------
// Config
// ----------------------------------------------------------------------------

type runtimeConfig struct {
	project     string
	environment string
	// vertexAILocation drives the GEMINI vendor adapter; live it is "global".
	vertexAILocation string
	// embedVertexLocation drives the EMBEDDING vendor adapter separately:
	// publisher embedding models (text-embedding-004) are NOT served at the
	// global endpoint, only regionally (ADR-163). Wiring embeds onto the
	// Gemini location 404s live (G1' gap 1, found 2026-08-13).
	embedVertexLocation      string
	modelArmorLocation       string
	grpcPort                 int
	httpPort                 int
	gatewayVersion           string
	observabilityDSNSecretID string // Secret Manager ID — DSN fetched at boot, fail-loud
	dbConnectTimeout         time.Duration
	identityGRPCAddr         string        // CHORA_IDENTITY_GRPC_ADDR — WS-1 mana metering port (fail-loud)
	manaStrictOnError        bool          // CHORA_MANA_STRICT_ON_ERROR — block fail-closed actions when the meter errors
	enforceContentsScreen    bool          // CHORA_ARMOR_ENFORCE_CONTENTS_SCREEN, G1'-3: block on the derived-contents Armor leg instead of auditing it
	enforceToolCallScreen    bool          // CHORA_ARMOR_ENFORCE_TOOL_CALL_SCREEN, G1'-2: block on the derived-tool-call Armor leg instead of auditing it
	enforceToolsScreen       bool          // CHORA_ARMOR_ENFORCE_TOOLS_SCREEN, CHO-2391: block on the derived tool-DECLARATION Armor leg instead of auditing it
	fallbackRetries          int           // CHORA_FALLBACK_RETRIES — extra same-provider attempts for a retryable failure (timeout / 429 / 5xx); 0 = advance to the next target
	fallbackRetryBackoff     time.Duration // CHORA_FALLBACK_RETRY_BACKOFF_MS — delay before each same-provider retry
	budgetRequired            bool          // CHORA_LLM_BUDGET_REQUIRED — fail-closed: a missing active budget window BLOCKS provider calls instead of allowing them (default false)
	vendorHTTPTimeout        time.Duration
	defaultTenantID          string // CHORA_DEFAULT_TENANT_ID — HTTP facade default tenant
	defaultGCID              string // CHORA_DEFAULT_GCID — HTTP facade default GCID
	defaultAgentID           string // CHORA_DEFAULT_AGENT_ID — HTTP facade default agent
}

func loadConfig() (runtimeConfig, error) {
	cfg := runtimeConfig{
		project:                  envOr("GCP_PROJECT", "chora-489812"),
		environment:              envOr("CHORA_ENVIRONMENT", "dev"),
		vertexAILocation:         envOr("VERTEX_AI_LOCATION", "asia-southeast1"),
		embedVertexLocation:      envOr("CHORA_EMBED_VERTEX_LOCATION", "asia-southeast1"),
		modelArmorLocation:       envOr("MODEL_ARMOR_LOCATION", "asia-southeast1"),
		grpcPort:                 envInt("CHORA_GRPC_PORT", 9090),
		httpPort:                 envInt("CHORA_HTTP_PORT", 8080),
		gatewayVersion:           envOr("SERVICE_VERSION", "chora-model-gateway:dev"),
		observabilityDSNSecretID: os.Getenv("CHORA_OBSERVABILITY_DSN_SECRET_ID"),
		identityGRPCAddr:         os.Getenv("CHORA_IDENTITY_GRPC_ADDR"),
		manaStrictOnError:        os.Getenv("CHORA_MANA_STRICT_ON_ERROR") == "true",
		enforceContentsScreen:    os.Getenv("CHORA_ARMOR_ENFORCE_CONTENTS_SCREEN") == "true",
		enforceToolCallScreen:    os.Getenv("CHORA_ARMOR_ENFORCE_TOOL_CALL_SCREEN") == "true",
		enforceToolsScreen:       os.Getenv("CHORA_ARMOR_ENFORCE_TOOLS_SCREEN") == "true",
		fallbackRetries:          envInt("CHORA_FALLBACK_RETRIES", 0),
		budgetRequired:            os.Getenv("CHORA_LLM_BUDGET_REQUIRED") == "true",
		defaultTenantID:          os.Getenv("CHORA_DEFAULT_TENANT_ID"),
		defaultGCID:              os.Getenv("CHORA_DEFAULT_GCID"),
	}
	if to := os.Getenv("CHORA_BOOTSTRAP_TIMEOUT_SECONDS"); to != "" {
		n, err := strconv.Atoi(to)
		if err != nil {
			return cfg, fmt.Errorf("CHORA_BOOTSTRAP_TIMEOUT_SECONDS invalid: %w", err)
		}
		cfg.dbConnectTimeout = time.Duration(n) * time.Second
	} else {
		cfg.dbConnectTimeout = 30 * time.Second
	}
	// Per-vendor-attempt HTTP budget for LLM calls. The 30s library default
	// times out awaiting first byte on large multimodal prompts (a 254KB
	// inline-PDF grounding call needs 30-60s TTFB on gemini-*-pro) — proven
	// live 2026-06-10 (Lane 1c W6): every >30s-TTFB attempt died with
	// "Client.Timeout exceeded while awaiting headers" and the whole batch
	// failed. Bound above by the orchestrator dispatch budget (httpx 120s).
	if to := os.Getenv("CHORA_VENDOR_HTTP_TIMEOUT_SECONDS"); to != "" {
		n, err := strconv.Atoi(to)
		if err != nil {
			return cfg, fmt.Errorf("CHORA_VENDOR_HTTP_TIMEOUT_SECONDS invalid: %w", err)
		}
		cfg.vendorHTTPTimeout = time.Duration(n) * time.Second
	} else {
		cfg.vendorHTTPTimeout = 110 * time.Second
	}
	// Same-provider retry budget for the FallbackEngine. A retryable failure
	// (timeout / 429 / 5xx) may be retried on the SAME provider this many times
	// before the walk advances to the next target. The default is 0: retrying a
	// rate-limited provider without an operator opting in amplifies the load
	// that caused the 429, so the matrix's "maybe" rows read as "no" by default.
	if to := os.Getenv("CHORA_FALLBACK_RETRY_BACKOFF_MS"); to != "" {
		n, err := strconv.Atoi(to)
		if err != nil {
			return cfg, fmt.Errorf("CHORA_FALLBACK_RETRY_BACKOFF_MS invalid: %w", err)
		}
		cfg.fallbackRetryBackoff = time.Duration(n) * time.Millisecond
	}
	if cfg.fallbackRetries < 0 {
		return cfg, fmt.Errorf("CHORA_FALLBACK_RETRIES must be >= 0")
	}
	cfg.defaultTenantID = envOr("CHORA_DEFAULT_TENANT_ID", "")
	cfg.defaultGCID = envOr("CHORA_DEFAULT_GCID", "")
	cfg.defaultAgentID = envOr("CHORA_DEFAULT_AGENT_ID", "openai_compat")
	// Validate non-empty required-by-env fields. observabilityDSNSecretID
	// is FAIL-LOUD per feedback_no_stubs_real_wiring — no fallback path,
	// no in-memory mode.
	for k, v := range map[string]string{
		"GCP_PROJECT":                       cfg.project,
		"CHORA_ENVIRONMENT":                 cfg.environment,
		"VERTEX_AI_LOCATION":                cfg.vertexAILocation,
		"CHORA_EMBED_VERTEX_LOCATION":       cfg.embedVertexLocation,
		"MODEL_ARMOR_LOCATION":              cfg.modelArmorLocation,
		"CHORA_OBSERVABILITY_DSN_SECRET_ID": cfg.observabilityDSNSecretID,
		// WS-1 umbrella metering is load-bearing (ADR-142 §4) — no silent
		// un-metered fallback (feedback_no_stubs_real_wiring). grpc.NewClient is
		// lazy, so a present-but-temporarily-down identity does not block boot.
		"CHORA_IDENTITY_GRPC_ADDR": cfg.identityGRPCAddr,
	} {
		if strings.TrimSpace(v) == "" {
			return cfg, fmt.Errorf("env %s required", k)
		}
	}
	return cfg, nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

// ----------------------------------------------------------------------------
// Token provider — Workload Identity Federation via google.DefaultTokenSource
// ----------------------------------------------------------------------------
//
// At runtime in GKE Autopilot, the in-cluster KSA→GSA binding (Phase 0.5 +
// Phase 0.6 IAM module) makes ADC return a TokenSource backed by the
// chora-model-gateway GSA. DefaultTokenSource caches refresh internally,
// so each .Token() call after the first is a cheap in-memory read until
// the underlying access token nears expiry.
//
// On a developer laptop with GOOGLE_APPLICATION_CREDENTIALS pointing at the
// dale-cli key file, ADC returns the dale-cli token source directly — the
// `gcloud auth application-default login` flow is NOT required.

type googleTokenProvider struct {
	src oauth2.TokenSource
}

func (g *googleTokenProvider) Token(ctx context.Context) (string, error) {
	if g == nil || g.src == nil {
		return "", fmt.Errorf("modelgateway: token source not initialised")
	}
	tok, err := g.src.Token()
	if err != nil {
		return "", fmt.Errorf("modelgateway: token mint: %w", err)
	}
	return tok.AccessToken, nil
}

func newTokenProvider(ctx context.Context) (*googleTokenProvider, error) {
	ts, err := google.DefaultTokenSource(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, fmt.Errorf("modelgateway: DefaultTokenSource: %w", err)
	}
	return &googleTokenProvider{src: ts}, nil
}

// ----------------------------------------------------------------------------
// Postgres open — FAIL-LOUD per feedback_no_stubs_real_wiring
// ----------------------------------------------------------------------------

// openRepo opens *sql.DB to chora_observability via the cloudsql-proxy
// sidecar at 127.0.0.1:5432. Any failure (sql.Open error, ping timeout,
// pg adapter construction error) returns a non-nil error so the caller
// exits non-zero and kubelet surfaces the misconfiguration as
// CrashLoopBackOff — no silent fallback to no-op repos.
//
// Retry loop on ping: cloudsql-proxy sidecar can be 5-15s slower than the
// service container to start (auth + private-ip TCP listener), and pgx
// returns immediately on connection-refused (it does NOT wait for ctx
// deadline like network-unreachable). Without this loop the service races
// the sidecar at boot and crashes on the first attempt — happens reliably
// on cold-start nodes, lucky-timing on warm ones. Loops every 500ms until
// `timeout` elapses or ping succeeds.
// openRepo returns the single *pg.Repo backing chora_observability — it
// implements BudgetRepo + OutboxWriter (Invoke) AND ExternalEgressGate +
// EgressAuditWriter (GroundedSearch, ADR-231). One handle, four ports.
func openRepo(dsn string, timeout time.Duration) (*pg.Repo, func(), error) {
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return nil, nil, fmt.Errorf("sql.Open: %w", err)
	}
	// Connection pool sizing — mirrors the Python reference
	// (asyncpg.create_pool min_size=1, max_size=20). The gateway is a
	// low-concurrency chokepoint (one in-flight LLM call per dispatch, a
	// handful of concurrent RPCs), so a small pool bounds the Postgres
	// connections each replica holds while still allowing bursts to 20.
	db.SetMaxOpenConns(20)
	db.SetMaxIdleConns(1)
	pingCtx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	var lastErr error
	for {
		// Each ping attempt gets its own 5s ctx so a long-hung dial doesn't
		// swallow the outer deadline. lastErr survives the loop so the
		// returned error still names the failure mode.
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
			return nil, nil, fmt.Errorf("db ping: %w (gave up after %s)", lastErr, timeout)
		case <-time.After(500 * time.Millisecond):
			// retry
		}
	}
	repo, err := pg.New(db)
	if err != nil {
		_ = db.Close()
		return nil, nil, err
	}
	cleanup := func() { _ = db.Close() }
	return repo, cleanup, nil
}

// ----------------------------------------------------------------------------
// Policy loader — Phase 2.4 in-memory default
// ----------------------------------------------------------------------------

type defaultPolicyLoader struct {
	armorTemplateStrict   string
	armorTemplateBalanced string
}

func newDefaultPolicyLoader(cfg runtimeConfig) *defaultPolicyLoader {
	return &defaultPolicyLoader{
		armorTemplateStrict: fmt.Sprintf("projects/%s/locations/%s/templates/chora-guardrail-strict-%s",
			cfg.project, cfg.modelArmorLocation, cfg.environment),
		armorTemplateBalanced: fmt.Sprintf("projects/%s/locations/%s/templates/chora-guardrail-balanced-%s",
			cfg.project, cfg.modelArmorLocation, cfg.environment),
	}
}

// ResolveAgentPolicy returns a sane default policy keyed on agent_id.
//
// AGENT-DRIVEN tiering (CR qgen 2026-06-01): the calling agent owns a
// build-time YAML declaring its primary + ordered fallback chain and sends
// the chain on InvokeRequest.fallback_logical_model_ids. The gateway HONOURS
// that declared chain rather than hardcoding a per-tier ladder here — each
// fallback id is dispatched against the SAME vendor the primary resolved to
// (Vertex AI Gemini for gemini-* ids). This IS the deferred "Phase-4 GCS-YAML
// policy" generalised to a request-scoped contract; armor-tier selection
// stays gateway-owned below.
func (l *defaultPolicyLoader) ResolveAgentPolicy(ctx context.Context, agentID, crewKind string, requested domain.LogicalModelID, fallbackModels []domain.LogicalModelID) (domain.AgentPolicy, error) {
	// Default policy: route to Vertex AI Gemini + balanced Armor template.
	policy := domain.AgentPolicy{
		AgentID:                agentID,
		ResolvedLogicalModelID: requested,
		Vendor:                 domain.VendorFamilyVertexGemini,
		ArmorTemplate:          l.armorTemplateBalanced,
	}
	// Honour the agent-declared fallback chain against the resolved vendor.
	for _, fm := range fallbackModels {
		if fm == "" {
			continue
		}
		policy.FallbackChain = append(policy.FallbackChain, domain.AgentPolicyFallback{
			Vendor:                 policy.Vendor,
			ResolvedLogicalModelID: fm,
		})
	}
	// Per-agent overrides: qgen + the kid-facing Companion chat use the strict
	// tier. THIS SWITCH IS THE ENFORCED TIER. chora-contracts/yaml/
	// agent-guardrail-mapping.yaml is copied into the image but never read
	// (its ENFORCED block says so); its default-strict is declaration only,
	// the enforced default here is BALANCED. So every renamed agent id MUST be
	// listed here or it silently drops to balanced: ADR-254 D9 renamed the
	// chat agent familiar_companion -> companion_chat (the live binary sends
	// "companion_chat"); familiar_companion stays until the old chora-familiar
	// Deployment is retired at G4. The other ADR-254 ids (companion_diagnoser,
	// companion_extractor, kg_explorer, qgen_renderer, content_recommender,
	// content_moderation, duel_atom_smith, profile_conjurer) keep their
	// predecessors' enforced BALANCED tier (the owner-ruled divergence
	// documented in the yaml). Pinned by cmd/server/policy_loader_test.go.
	switch agentID {
	case "qgen_question", "qgen_critic", "familiar_companion", "companion_chat":
		policy.ArmorTemplate = l.armorTemplateStrict
	}
	return policy, nil
}

// ----------------------------------------------------------------------------
// UUIDv7 generator for invocation_id default
// ----------------------------------------------------------------------------

// newInvocationID returns a UUIDv7 string via github.com/google/uuid v1.6.0.
// UUIDv7 is time-ordered (Unix-millisecond prefix + 74 bits of randomness),
// so ledger dedup at the gateway's invocation-id slot stays roughly sorted
// without requiring a CLUSTERED INDEX on time.
//
// uuid.NewV7() can in principle fail if crypto/rand is exhausted; we fall
// back to v4 in that case to keep the request flowing (the ledger uniqueness
// constraint is what actually guards against dedup, not UUID version).
func newInvocationID() string {
	id, err := uuid.NewV7()
	if err != nil {
		return uuid.NewString()
	}
	return id.String()
}
