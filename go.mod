module github.com/5007-Capstone/chora/services/chora-model-gateway

go 1.26.1

require (
	github.com/apollo-chora/chora-contracts/gen/go v0.0.0-20261006160207-6eeb884368e2
	github.com/jackc/pgx/v5 v5.9.2
	github.com/stretchr/testify v1.11.1
	go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc v0.68.0
	go.opentelemetry.io/otel v1.44.0
	go.opentelemetry.io/otel/trace v1.44.0
	google.golang.org/grpc v1.83.0
	google.golang.org/protobuf v1.36.11
)

require (
	cloud.google.com/go/auth v0.20.0 // indirect
	cloud.google.com/go/auth/oauth2adapt v0.2.8 // indirect
	cloud.google.com/go/compute/metadata v0.9.0 // indirect
	cloud.google.com/go/logging v1.18.0 // indirect
	cloud.google.com/go/longrunning v1.0.0 // indirect
	cloud.google.com/go/monitoring v1.29.0 // indirect
	cloud.google.com/go/trace v1.16.0 // indirect
	github.com/GoogleCloudPlatform/opentelemetry-operations-go/exporter/trace v1.32.0 // indirect
	github.com/GoogleCloudPlatform/opentelemetry-operations-go/internal/resourcemapping v0.57.0 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/felixge/httpsnoop v1.0.4 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/google/s2a-go v0.1.9 // indirect
	github.com/googleapis/enterprise-certificate-proxy v0.3.16 // indirect
	github.com/googleapis/gax-go/v2 v2.22.0 // indirect
	go.opentelemetry.io/auto/sdk v1.2.1 // indirect
	go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp v0.67.0 // indirect
	go.opentelemetry.io/otel/exporters/stdout/stdouttrace v1.43.0 // indirect
	go.opentelemetry.io/otel/metric v1.44.0 // indirect
	go.opentelemetry.io/otel/sdk v1.44.0 // indirect
	golang.org/x/crypto v0.55.0 // indirect
	golang.org/x/time v0.15.0 // indirect
	google.golang.org/api v0.285.0 // indirect
	google.golang.org/genproto/googleapis/api v0.0.0-20260615183401-62b3387ff324 // indirect
)

// Resolved through go.work at build time. The replace directive lets
// `go test` / `go build` from this module dir alone (outside workspace
// mode) still find the generated stubs.
replace github.com/apollo-chora/chora-contracts/gen/go => ../chora-contracts/gen/go

require (
	github.com/5007-Capstone/chora/libs/chora-go-common v0.0.0-00010101000000-000000000000
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc // indirect
	github.com/google/uuid v1.6.0
	github.com/jackc/pgpassfile v1.0.0 // indirect
	github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761 // indirect
	github.com/jackc/puddle/v2 v2.2.2 // indirect
	github.com/pmezard/go-difflib v1.0.1-0.20181226105442-5d4384ee4fb2 // indirect
	golang.org/x/net v0.57.0 // indirect
	golang.org/x/oauth2 v0.36.0
	golang.org/x/sync v0.22.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
	golang.org/x/text v0.41.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260615183401-62b3387ff324 // indirect
	gopkg.in/yaml.v3 v3.0.1
)

replace github.com/5007-Capstone/chora/libs/chora-go-common => ../chora-base/libs/chora-go-common
