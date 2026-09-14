package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func TestMapBedrockModel(t *testing.T) {
	t.Setenv("BEDROCK_MODEL", "sol")
	t.Setenv("BEDROCK_SMALL_MODEL", "terra")
	if got := mapBedrockModel("claude-opus-5"); got != "us.openai.gpt-5.6-sol" {
		t.Fatalf("main model = %q", got)
	}
	if got := mapBedrockModel("claude-haiku-4-5"); got != "us.openai.gpt-5.6-terra" {
		t.Fatalf("small model = %q", got)
	}
	t.Setenv("BEDROCK_MODEL", "global.openai.gpt-6-astra")
	if got := mapBedrockModel("anything"); got != "global.openai.gpt-6-astra" {
		t.Fatalf("full model id = %q", got)
	}
}

func TestResolveBedrockEffort(t *testing.T) {
	req := &anthropicRequest{}
	if got := resolveBedrockEffort(req, "us.openai.gpt-6-astra"); got != "high" {
		t.Fatalf("astra default = %q", got)
	}
	if got := resolveBedrockEffort(req, "us.openai.gpt-5.6-sol"); got != "high" {
		t.Fatalf("sol default = %q", got)
	}
	if got := resolveBedrockEffort(req, "us.openai.gpt-5.6-terra"); got != "max" {
		t.Fatalf("terra default = %q", got)
	}
	if got := resolveBedrockEffort(req, "us.openai.gpt-5.6-luna"); got != "medium" {
		t.Fatalf("luna default = %q", got)
	}
	t.Setenv("CLAUDE_CODE_EFFORT_LEVEL", "low")
	if got := resolveBedrockEffort(req, "us.openai.gpt-6-astra"); got != "low" {
		t.Fatalf("Claude Code effort = %q", got)
	}
	req.Effort = "high"
	if got := resolveBedrockEffort(req, "us.openai.gpt-6-astra"); got != "high" {
		t.Fatalf("request effort = %q", got)
	}
	t.Setenv("BEDROCK_EFFORT", "medium")
	if got := resolveBedrockEffort(req, "us.openai.gpt-6-astra"); got != "medium" {
		t.Fatalf("BEDROCK_EFFORT precedence = %q", got)
	}
}

func TestTranslateBedrockRequest(t *testing.T) {
	t.Setenv("BEDROCK_MODEL", "terra")
	t.Setenv("BEDROCK_EFFORT", "max")
	temperature, topP := 0.4, 0.8
	req := &anthropicRequest{
		Model: "claude-opus-5", MaxTokens: 321, System: json.RawMessage(`[{"type":"text","text":"system"}]`),
		Temperature: &temperature, TopP: &topP, StopSequences: []string{"STOP"},
		Tools:      []anthropicTool{{Name: "Bash", Description: "Run a command", InputSchema: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}}}`)}},
		ToolChoice: json.RawMessage(`{"type":"tool","name":"Bash"}`),
		Messages: []anthropicMessage{
			{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"toolu_1","name":"Bash","input":{"command":"ls"}}]`)},
			{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"toolu_1","content":{"files":["a"]},"is_error":true}]`)},
		},
	}
	out, err := translateBedrockRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if aws.ToString(out.ModelId) != "us.openai.gpt-5.6-terra" || aws.ToInt32(out.InferenceConfig.MaxTokens) != 321 {
		t.Fatalf("model/config: %+v", out)
	}
	if len(out.InferenceConfig.StopSequences) != 1 || out.InferenceConfig.StopSequences[0] != "STOP" {
		t.Fatalf("stop sequences: %+v", out.InferenceConfig.StopSequences)
	}
	if out.InferenceConfig.Temperature != nil || out.InferenceConfig.TopP != nil {
		t.Fatal("temperature and top_p must be omitted for OpenAI Bedrock models")
	}
	fields, err := out.AdditionalModelRequestFields.MarshalSmithyDocument()
	if err != nil || string(fields) != `{"reasoning":{"effort":"max"}}` {
		t.Fatalf("additional request fields = %s, err = %v", fields, err)
	}
	if len(out.System) != 1 || len(out.Messages) != 2 || out.ToolConfig == nil || len(out.ToolConfig.Tools) != 1 {
		t.Fatalf("translated shape: %+v", out)
	}
	toolUse := out.Messages[0].Content[0].(*brtypes.ContentBlockMemberToolUse).Value
	if toolUse.Input == nil {
		t.Fatal("tool input document is nil")
	}

	toolResult := out.Messages[1].Content[0].(*brtypes.ContentBlockMemberToolResult).Value
	if toolResult.Status != brtypes.ToolResultStatusError {
		t.Fatalf("tool status = %q", toolResult.Status)
	}
	jsonResult := toolResult.Content[0].(*brtypes.ToolResultContentBlockMemberJson).Value
	if jsonResult == nil {
		t.Fatal("tool result JSON document is nil")
	}
}

func TestTranslateBedrockImage(t *testing.T) {
	msg, err := translateBedrockMessage(anthropicMessage{Role: "user", Content: json.RawMessage(`[
		{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}},
		{"type":"text","text":"describe"}
	]`)})
	if err != nil {
		t.Fatal(err)
	}
	if len(msg.Content) != 2 {
		t.Fatalf("content: %+v", msg.Content)
	}
	image := msg.Content[0].(*brtypes.ContentBlockMemberImage).Value
	if image.Format != brtypes.ImageFormatPng {
		t.Fatalf("image format = %q", image.Format)
	}
	bytes := image.Source.(*brtypes.ImageSourceMemberBytes).Value
	if string(bytes) != "hello" {
		t.Fatalf("image bytes = %q", bytes)
	}
}

func TestBedrockCountTokensIsMarkedAsEstimate(t *testing.T) {
	request := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader("12345678"))
	response := httptest.NewRecorder()
	(&bedrockServer{}).mux().ServeHTTP(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status = %d", response.Code)
	}
	if got := response.Header().Get("X-Claude2OpenAI-Token-Count"); got != "estimate" {
		t.Fatalf("estimate header = %q", got)
	}
	var body map[string]int
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	if body["input_tokens"] != 2 {
		t.Fatalf("input_tokens = %d", body["input_tokens"])
	}
}

func TestBedrockTranslationErrorUsesHTTPStatusBeforeStreamStarts(t *testing.T) {
	body := `{"model":"claude-opus-4-6","max_tokens":10,"stream":true,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"!!!"}}]}]}`
	request := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(body))
	response := httptest.NewRecorder()
	(&bedrockServer{}).mux().ServeHTTP(response, request)

	if response.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, body = %s", response.Code, response.Body.String())
	}
	if got := response.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("content type = %q", got)
	}
	var payload struct {
		Type  string `json:"type"`
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &payload); err != nil {
		t.Fatal(err)
	}
	if payload.Type != "error" || payload.Error.Type != "invalid_request_error" {
		t.Fatalf("error payload = %+v", payload)
	}
}

func TestBedrockStreamSequence(t *testing.T) {
	state := newBedrockStreamState("us.openai.gpt-6-astra")
	events := []brtypes.ConverseStreamOutput{
		&brtypes.ConverseStreamOutputMemberMessageStart{Value: brtypes.MessageStartEvent{Role: brtypes.ConversationRoleAssistant}},
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(0), Delta: &brtypes.ContentBlockDeltaMemberText{Value: "hello"}}},
		&brtypes.ConverseStreamOutputMemberContentBlockStop{Value: brtypes.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(0)}},
		&brtypes.ConverseStreamOutputMemberContentBlockStart{Value: brtypes.ContentBlockStartEvent{ContentBlockIndex: aws.Int32(1), Start: &brtypes.ContentBlockStartMemberToolUse{Value: brtypes.ToolUseBlockStart{ToolUseId: aws.String("toolu_1"), Name: aws.String("Bash")}}}},
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(1), Delta: &brtypes.ContentBlockDeltaMemberToolUse{Value: brtypes.ToolUseBlockDelta{Input: aws.String(`{"command":"ls"}`)}}}},
		&brtypes.ConverseStreamOutputMemberContentBlockStop{Value: brtypes.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(1)}},
		&brtypes.ConverseStreamOutputMemberMessageStop{Value: brtypes.MessageStopEvent{StopReason: brtypes.StopReasonToolUse}},
		&brtypes.ConverseStreamOutputMemberMetadata{Value: brtypes.ConverseStreamMetadataEvent{Metrics: &brtypes.ConverseStreamMetrics{}, Usage: &brtypes.TokenUsage{InputTokens: aws.Int32(10), OutputTokens: aws.Int32(4), TotalTokens: aws.Int32(14), CacheReadInputTokens: aws.Int32(3), CacheWriteInputTokens: aws.Int32(2)}}},
	}
	var out []anthropicEvent
	for _, event := range events {
		out = append(out, state.handle(event)...)
	}
	out = append(out, state.finalize()...)
	want := "message_start,content_block_start,content_block_delta,content_block_stop,content_block_start,content_block_delta,content_block_stop,message_delta,message_stop"
	if got := strings.Join(eventTypes(out), ","); got != want {
		t.Fatalf("event sequence:\n got %s\nwant %s", got, want)
	}
	msg, err := aggregateEvents(out, "us.openai.gpt-6-astra")
	if err != nil {
		t.Fatal(err)
	}
	if msg.StopReason != "tool_use" || msg.Usage["input_tokens"] != 10 || msg.Usage["cache_read_input_tokens"] != 3 || msg.Usage["cache_creation_input_tokens"] != 2 {
		t.Fatalf("metadata: stop=%s usage=%+v", msg.StopReason, msg.Usage)
	}
	if len(msg.Content) != 2 || msg.Content[0]["text"] != "hello" || msg.Content[1]["type"] != "tool_use" {
		t.Fatalf("content: %+v", msg.Content)
	}
}

// A redacted reasoning block at Bedrock index 0 must not leak a gap: the text
// that follows at index 1 is the client's block 0, otherwise the unary
// aggregation drops it (seen on gpt-5.6-terra at effort max, 14-sep-2026).
func TestBedrockStreamRenumbersAfterReasoning(t *testing.T) {
	state := newBedrockStreamState("us.openai.gpt-5.6-terra")
	events := []brtypes.ConverseStreamOutput{
		&brtypes.ConverseStreamOutputMemberMessageStart{Value: brtypes.MessageStartEvent{Role: brtypes.ConversationRoleAssistant}},
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(0), Delta: &brtypes.ContentBlockDeltaMemberReasoningContent{Value: &brtypes.ReasoningContentBlockDeltaMemberRedactedContent{Value: []byte("x")}}}},
		&brtypes.ConverseStreamOutputMemberContentBlockStop{Value: brtypes.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(0)}},
		&brtypes.ConverseStreamOutputMemberContentBlockDelta{Value: brtypes.ContentBlockDeltaEvent{ContentBlockIndex: aws.Int32(1), Delta: &brtypes.ContentBlockDeltaMemberText{Value: "BEDROCK OPENAI OK"}}},
		&brtypes.ConverseStreamOutputMemberContentBlockStop{Value: brtypes.ContentBlockStopEvent{ContentBlockIndex: aws.Int32(1)}},
		&brtypes.ConverseStreamOutputMemberMessageStop{Value: brtypes.MessageStopEvent{StopReason: brtypes.StopReasonEndTurn}},
	}
	var out []anthropicEvent
	for _, event := range events {
		out = append(out, state.handle(event)...)
	}
	out = append(out, state.finalize()...)
	want := "message_start,content_block_start,content_block_delta,content_block_stop,message_delta,message_stop"
	if got := strings.Join(eventTypes(out), ","); got != want {
		t.Fatalf("event sequence:\n got %s\nwant %s", got, want)
	}
	msg, err := aggregateEvents(out, "us.openai.gpt-5.6-terra")
	if err != nil {
		t.Fatal(err)
	}
	if len(msg.Content) != 1 || msg.Content[0]["text"] != "BEDROCK OPENAI OK" {
		t.Fatalf("content = %+v", msg.Content)
	}
}
