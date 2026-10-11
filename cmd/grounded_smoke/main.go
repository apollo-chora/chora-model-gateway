// Temporary GroundedSearch smoke client. Not part of the service.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	model_gatewayv1 "github.com/apollo-chora/chora-contracts/gen/go/chora/services/model_gateway/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

func main() {
	addr := os.Args[1]
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("dial: %v", err)
	}
	defer func() { _ = conn.Close() }()

	client := model_gatewayv1.NewModelGatewayServiceClient(conn)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	resp, err := client.GroundedSearch(ctx, &model_gatewayv1.GroundedSearchRequest{
		TenantId:   "00000000-0000-7000-8000-000000000001",
		Gcid:       "00000000-0000-7000-8000-000000000002",
		AgentId:    "companion_seeker",
		Directive:  "What is the capital of France?",
		ActionCode: "grounded_search_smoke",
		MaxResults: 5,
	})
	if err != nil {
		log.Fatalf("GroundedSearch: %v", err)
	}
	fmt.Printf("vendor=%s model=%s finish=%s\n", resp.GetVendor(), resp.GetModelVersion(), resp.GetFinishReason())
	fmt.Printf("answer: %s\n", resp.GetGroundedAnswer())
	for _, c := range resp.GetCitations() {
		fmt.Printf("citation: %s | %s | %s (%.2f)\n", c.GetTitle(), c.GetUri(), c.GetDomain(), c.GetConfidence())
	}
}
