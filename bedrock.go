package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	brtypes "github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/aws/smithy-go"
)

const defaultBedrockPort = 3458

var thinkingDropOnce sync.Once

type bedrockConverseClient interface {
	ConverseStream(context.Context, *bedrockruntime.ConverseStreamInput, ...func(*bedrockruntime.Options)) (*bedrockruntime.ConverseStreamOutput, error)
}

type bedrockServer struct {
	client bedrockConverseClient
}

func bedrockEnabled() bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv("BEDROCK_OPENAI")))
	return v == "1" || v == "true" || v == "yes"
}

func bedrockModelAlias(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", "astra":
		return "us.openai.gpt-6-astra"
	case "sol":
		return "us.openai.gpt-5.6-sol"
	case "terra":
		return "us.openai.gpt-5.6-terra"
	case "luna":
		return "us.openai.gpt-5.6-luna"
	default:
		return strings.TrimSpace(value)
	}
}

func mapBedrockModel(requested string) string {
	if strings.Contains(strings.ToLower(requested), "haiku") {
		small := strings.TrimSpace(os.Getenv("BEDROCK_SMALL_MODEL"))
		if small == "" {
			small = "luna"
		}
		return bedrockModelAlias(small)
	}
	return bedrockModelAlias(os.Getenv("BEDROCK_MODEL"))
}

func bedrockEffortDefault(model string) string {
	switch {
	case strings.Contains(model, "gpt-5.6-terra"):
		return "max"
	case strings.Contains(model, "gpt-5.6-luna"):
		return "medium"
	default:
		return "high"
	}
}

func isBedrockEffort(value string) bool {
	switch value {
	case "low", "medium", "high", "max":
		return true
	default:
		return false
	}
}

func claudeCodeEffort(req *anthropicRequest) string {
	if req.Effort != "" {
		return strings.ToLower(strings.TrimSpace(req.Effort))
	}
	if req.OutputConfig != nil && req.OutputConfig.Effort != "" {
		return strings.ToLower(strings.TrimSpace(req.OutputConfig.Effort))
	}
	if value := strings.TrimSpace(os.Getenv("CLAUDE_CODE_EFFORT_LEVEL")); value != "" {
		return strings.ToLower(value)
	}
	if value, ok := thinkingEffort(req.Thinking); ok {
		return value
	}
	return ""
}

func resolveBedrockEffort(req *anthropicRequest, model string) string {
	if value := strings.ToLower(strings.TrimSpace(os.Getenv("BEDROCK_EFFORT"))); value != "" {
		if isBedrockEffort(value) {
			return value
		}
		log.Printf("bedrock: ignoring invalid BEDROCK_EFFORT %q; expected low, medium, high, or max", value)
	}
	if value := claudeCodeEffort(req); value != "" {
		if value == "xhigh" {
			return "high"
		}
		if isBedrockEffort(value) {
			return value
		}
		log.Printf("bedrock: ignoring unsupported Claude Code effort %q", value)
	}
	return bedrockEffortDefault(model)
}

func loadBedrockClient(ctx context.Context) (bedrockConverseClient, error) {
	profile := strings.TrimSpace(os.Getenv("AWS_PROFILE"))
	region := strings.TrimSpace(os.Getenv("AWS_REGION"))
	if region == "" {
		region = strings.TrimSpace(os.Getenv("AWS_DEFAULT_REGION"))
	}
	if region == "" {
		region = "us-west-2"
	}
	opts := []func(*awsconfig.LoadOptions) error{awsconfig.WithRegion(region)}
	if profile != "" {
		opts = append(opts, awsconfig.WithSharedConfigProfile(profile))
	}
	cfg, err := awsconfig.LoadDefaultConfig(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("cannot load AWS profile %q in %s: %w", profile, region, err)
	}
	if _, err := cfg.Credentials.Retrieve(ctx); err != nil {
		return nil, fmt.Errorf("AWS credentials unavailable for profile %q: %w; run `aws sso login --sso-session dfx5`", profile, err)
	}
	return bedrockruntime.NewFromConfig(cfg), nil
}

func (s *bedrockServer) mux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Claude2OpenAI-Backend", "bedrock")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	mux.HandleFunc("POST /v1/messages", s.handleMessages)
	mux.HandleFunc("POST /v1/messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
		if err != nil {
			writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "cannot read request body", false)
			return
		}
		n := len(body) / 4
		if n < 1 {
			n = 1
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Claude2OpenAI-Token-Count", "estimate")
		_ = json.NewEncoder(w).Encode(map[string]any{"input_tokens": n})
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeAnthropicError(w, http.StatusNotFound, "not_found_error", "unknown endpoint: "+r.URL.Path, false)
	})
	return mux
}

func translateBedrockRequest(req *anthropicRequest) (*bedrockruntime.ConverseStreamInput, error) {
	model := mapBedrockModel(req.Model)
	maxTokens := int32(req.MaxTokens)
	if maxTokens <= 0 {
		maxTokens = 8192
	}
	effort := resolveBedrockEffort(req, model)
	out := &bedrockruntime.ConverseStreamInput{
		ModelId: aws.String(model),
		InferenceConfig: &brtypes.InferenceConfiguration{
			MaxTokens:     &maxTokens,
			StopSequences: append([]string(nil), req.StopSequences...),
		},
		AdditionalModelRequestFields: document.NewLazyDocument(map[string]any{"reasoning": map[string]string{"effort": effort}}),
	}
	for _, b := range parseBlocks(req.System) {
		if b.Type == "text" && b.Text != "" {
			out.System = append(out.System, &brtypes.SystemContentBlockMemberText{Value: b.Text})
		}
	}
	for _, m := range req.Messages {
		translated, err := translateBedrockMessage(m)
		if err != nil {
			return nil, err
		}
		if len(translated.Content) > 0 {
			out.Messages = append(out.Messages, translated)
		}
	}
	if len(req.Tools) > 0 {
		tools, err := translateBedrockTools(req.Tools)
		if err != nil {
			return nil, err
		}
		out.ToolConfig = &brtypes.ToolConfiguration{Tools: tools, ToolChoice: translateBedrockToolChoice(req.ToolChoice)}
	}
	if req.Thinking != nil && req.Thinking.Type == "enabled" {
		thinkingDropOnce.Do(func() {
			log.Printf("bedrock: Anthropic thinking blocks are not supported by OpenAI Converse models and are dropped")
		})
	}
	return out, nil
}

func translateBedrockMessage(m anthropicMessage) (brtypes.Message, error) {
	role := brtypes.ConversationRoleUser
	if m.Role == "assistant" {
		role = brtypes.ConversationRoleAssistant
	}
	out := brtypes.Message{Role: role}
	for _, b := range parseBlocks(m.Content) {
		switch b.Type {
		case "text":
			out.Content = append(out.Content, &brtypes.ContentBlockMemberText{Value: b.Text})
		case "image":
			if b.Source == nil || b.Source.Type != "base64" || b.Source.Data == "" {
				continue
			}
			data, err := base64.StdEncoding.DecodeString(b.Source.Data)
			if err != nil {
				return out, fmt.Errorf("invalid base64 image: %w", err)
			}
			format, err := bedrockImageFormat(b.Source.MediaType)
			if err != nil {
				return out, err
			}
			out.Content = append(out.Content, &brtypes.ContentBlockMemberImage{Value: brtypes.ImageBlock{
				Format: format,
				Source: &brtypes.ImageSourceMemberBytes{Value: data},
			}})
		case "tool_use":
			var input any = map[string]any{}
			if len(b.Input) > 0 && string(b.Input) != "null" {
				if err := json.Unmarshal(b.Input, &input); err != nil {
					return out, fmt.Errorf("invalid tool input for %s: %w", b.Name, err)
				}
			}
			out.Content = append(out.Content, &brtypes.ContentBlockMemberToolUse{Value: brtypes.ToolUseBlock{
				ToolUseId: aws.String(b.ID), Name: aws.String(sanitizeToolName(b.Name)), Input: document.NewLazyDocument(input),
			}})
		case "tool_result":
			content := translateBedrockToolResultContent(b.Content)
			status := brtypes.ToolResultStatusSuccess
			if b.IsError {
				status = brtypes.ToolResultStatusError
			}
			out.Content = append(out.Content, &brtypes.ContentBlockMemberToolResult{Value: brtypes.ToolResultBlock{
				ToolUseId: aws.String(b.ToolUseID), Status: status, Content: content,
			}})
		case "thinking", "redacted_thinking":
			thinkingDropOnce.Do(func() { log.Printf("bedrock: thinking history is dropped for OpenAI Converse models") })
		}
	}
	return out, nil
}

func translateBedrockToolResultContent(raw json.RawMessage) []brtypes.ToolResultContentBlock {
	var value any
	if len(raw) > 0 && json.Unmarshal(raw, &value) == nil {
		switch v := value.(type) {
		case string:
			return []brtypes.ToolResultContentBlock{&brtypes.ToolResultContentBlockMemberText{Value: v}}
		case map[string]any:
			return []brtypes.ToolResultContentBlock{&brtypes.ToolResultContentBlockMemberJson{Value: document.NewLazyDocument(v)}}
		}
	}
	text := toolResultText(raw)
	if text == "" {
		text = "(no output)"
	}
	return []brtypes.ToolResultContentBlock{&brtypes.ToolResultContentBlockMemberText{Value: text}}
}

func bedrockImageFormat(mediaType string) (brtypes.ImageFormat, error) {
	switch strings.ToLower(strings.TrimSpace(mediaType)) {
	case "image/png":
		return brtypes.ImageFormatPng, nil
	case "image/jpeg", "image/jpg":
		return brtypes.ImageFormatJpeg, nil
	case "image/gif":
		return brtypes.ImageFormatGif, nil
	case "image/webp":
		return brtypes.ImageFormatWebp, nil
	default:
		return "", fmt.Errorf("unsupported image media type %q", mediaType)
	}
}

func translateBedrockTools(in []anthropicTool) ([]brtypes.Tool, error) {
	out := make([]brtypes.Tool, 0, len(in))
	for _, t := range in {
		var schema any = map[string]any{"type": "object", "properties": map[string]any{}}
		if len(t.InputSchema) > 0 {
			if err := json.Unmarshal(t.InputSchema, &schema); err != nil {
				return nil, fmt.Errorf("invalid input schema for tool %s: %w", t.Name, err)
			}
		}
		out = append(out, &brtypes.ToolMemberToolSpec{Value: brtypes.ToolSpecification{
			Name: aws.String(sanitizeToolName(t.Name)), Description: aws.String(t.Description),
			InputSchema: &brtypes.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(schema)},
		}})
	}
	return out, nil
}

func translateBedrockToolChoice(raw json.RawMessage) brtypes.ToolChoice {
	if len(raw) == 0 {
		return &brtypes.ToolChoiceMemberAuto{Value: brtypes.AutoToolChoice{}}
	}
	var bare string
	if json.Unmarshal(raw, &bare) == nil {
		switch bare {
		case "any":
			return &brtypes.ToolChoiceMemberAny{Value: brtypes.AnyToolChoice{}}
		default:
			return &brtypes.ToolChoiceMemberAuto{Value: brtypes.AutoToolChoice{}}
		}
	}
	var tc struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if json.Unmarshal(raw, &tc) != nil {
		return &brtypes.ToolChoiceMemberAuto{Value: brtypes.AutoToolChoice{}}
	}
	switch tc.Type {
	case "any":
		return &brtypes.ToolChoiceMemberAny{Value: brtypes.AnyToolChoice{}}
	case "tool":
		if tc.Name != "" {
			return &brtypes.ToolChoiceMemberTool{Value: brtypes.SpecificToolChoice{Name: aws.String(sanitizeToolName(tc.Name))}}
		}
		return &brtypes.ToolChoiceMemberAny{Value: brtypes.AnyToolChoice{}}
	default:
		return &brtypes.ToolChoiceMemberAuto{Value: brtypes.AutoToolChoice{}}
	}
}

func (s *bedrockServer) handleMessages(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "cannot read request body", false)
		return
	}
	var req anthropicRequest
	if err := json.Unmarshal(body, &req); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "invalid JSON: "+err.Error(), false)
		return
	}
	input, err := translateBedrockRequest(&req)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error(), false)
		return
	}
	log.Printf("bedrock messages: model %q -> %q, effort=%q, %d messages, stream=%v", req.Model, aws.ToString(input.ModelId), resolveBedrockEffort(&req, aws.ToString(input.ModelId)), len(input.Messages), req.Stream)
	response, err := s.converseWithRetry(r.Context(), input)
	if err != nil {
		status, errType, message := bedrockError(err)
		writeAnthropicError(w, status, errType, message, false)
		return
	}
	defer response.GetStream().Close()
	if req.Stream {
		s.streamResponse(w, response.GetStream(), aws.ToString(input.ModelId))
		return
	}
	s.unaryResponse(w, response.GetStream(), aws.ToString(input.ModelId))
}

func (s *bedrockServer) converseWithRetry(ctx context.Context, input *bedrockruntime.ConverseStreamInput) (*bedrockruntime.ConverseStreamOutput, error) {
	var out *bedrockruntime.ConverseStreamOutput
	var err error
	for attempt := 0; attempt < 3; attempt++ {
		out, err = s.client.ConverseStream(ctx, input)
		if err == nil || !isBedrockThrottle(err) || attempt == 2 {
			return out, err
		}
		timer := time.NewTimer(time.Duration(250*(1<<attempt)) * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
	return out, err
}

func isBedrockThrottle(err error) bool {
	var apiErr smithy.APIError
	return errors.As(err, &apiErr) && (apiErr.ErrorCode() == "ThrottlingException" || apiErr.ErrorCode() == "TooManyRequestsException")
}

func bedrockError(err error) (int, string, string) {
	message := sanitizeUpstreamError(err)
	var apiErr smithy.APIError
	if errors.As(err, &apiErr) {
		switch apiErr.ErrorCode() {
		case "ThrottlingException", "TooManyRequestsException":
			return http.StatusTooManyRequests, "rate_limit_error", message
		case "ValidationException":
			return http.StatusBadRequest, "invalid_request_error", message
		case "AccessDeniedException", "UnrecognizedClientException":
			return http.StatusForbidden, "permission_error", message
		}
	}
	lower := strings.ToLower(message)
	if strings.Contains(lower, "sso") && (strings.Contains(lower, "expired") || strings.Contains(lower, "token")) {
		message += "; run `aws sso login --sso-session dfx5`"
		return http.StatusUnauthorized, "authentication_error", message
	}
	return http.StatusBadGateway, "api_error", message
}

type bedrockStreamState struct {
	model       string
	messageID   string
	started     bool
	stopReason  string
	usage       map[string]int
	blockKinds  map[int32]string
	blockOpened map[int32]bool
	finished    bool
}

func newBedrockStreamState(model string) *bedrockStreamState {
	return &bedrockStreamState{
		model: model, messageID: "msg_" + randomHex(12), stopReason: "end_turn",
		usage:      map[string]int{"input_tokens": 0, "output_tokens": 0},
		blockKinds: map[int32]string{}, blockOpened: map[int32]bool{},
	}
}

func (s *bedrockStreamState) handle(event brtypes.ConverseStreamOutput) []anthropicEvent {
	var out []anthropicEvent
	start := func() {
		if !s.started {
			s.started = true
			out = append(out, messageStartEvent(s.messageID, s.model))
		}
	}
	switch ev := event.(type) {
	case *brtypes.ConverseStreamOutputMemberMessageStart:
		start()
	case *brtypes.ConverseStreamOutputMemberContentBlockStart:
		start()
		idx := aws.ToInt32(ev.Value.ContentBlockIndex)
		if tool, ok := ev.Value.Start.(*brtypes.ContentBlockStartMemberToolUse); ok {
			s.blockKinds[idx] = "tool_use"
			s.blockOpened[idx] = true
			out = append(out, blockStartEvent(int(idx), map[string]any{
				"type": "tool_use", "id": aws.ToString(tool.Value.ToolUseId), "name": aws.ToString(tool.Value.Name), "input": map[string]any{},
			}))
		}
	case *brtypes.ConverseStreamOutputMemberContentBlockDelta:
		start()
		idx := aws.ToInt32(ev.Value.ContentBlockIndex)
		switch delta := ev.Value.Delta.(type) {
		case *brtypes.ContentBlockDeltaMemberText:
			if !s.blockOpened[idx] {
				s.blockKinds[idx], s.blockOpened[idx] = "text", true
				out = append(out, blockStartEvent(int(idx), map[string]any{"type": "text", "text": ""}))
			}
			out = append(out, blockDeltaEvent(int(idx), map[string]any{"type": "text_delta", "text": delta.Value}))
		case *brtypes.ContentBlockDeltaMemberToolUse:
			out = append(out, blockDeltaEvent(int(idx), map[string]any{"type": "input_json_delta", "partial_json": aws.ToString(delta.Value.Input)}))
		}
	case *brtypes.ConverseStreamOutputMemberContentBlockStop:
		idx := aws.ToInt32(ev.Value.ContentBlockIndex)
		if s.blockOpened[idx] {
			s.blockOpened[idx] = false
			out = append(out, blockStopEvent(int(idx)))
		}
	case *brtypes.ConverseStreamOutputMemberMessageStop:
		start()
		s.stopReason = mapBedrockStopReason(ev.Value.StopReason)
	case *brtypes.ConverseStreamOutputMemberMetadata:
		start()
		if usage := ev.Value.Usage; usage != nil {
			s.usage["input_tokens"] = int(aws.ToInt32(usage.InputTokens))
			s.usage["output_tokens"] = int(aws.ToInt32(usage.OutputTokens))
			if usage.CacheReadInputTokens != nil {
				s.usage["cache_read_input_tokens"] = int(aws.ToInt32(usage.CacheReadInputTokens))
			}
			if usage.CacheWriteInputTokens != nil {
				s.usage["cache_creation_input_tokens"] = int(aws.ToInt32(usage.CacheWriteInputTokens))
			}
		}
		s.finished = true
		out = append(out, s.finish()...)
	}
	return out
}

func (s *bedrockStreamState) finish() []anthropicEvent {
	return []anthropicEvent{
		{"message_delta", map[string]any{
			"type": "message_delta", "delta": map[string]any{"stop_reason": s.stopReason, "stop_sequence": nil}, "usage": s.usage,
		}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}
}

func (s *bedrockStreamState) finalize() []anthropicEvent {
	if s.finished {
		return nil
	}
	var out []anthropicEvent
	if !s.started {
		s.started = true
		out = append(out, messageStartEvent(s.messageID, s.model))
	}
	for idx, opened := range s.blockOpened {
		if opened {
			out = append(out, blockStopEvent(int(idx)))
		}
	}
	out = append(out, s.finish()...)
	return out
}

func mapBedrockStopReason(reason brtypes.StopReason) string {
	switch reason {
	case brtypes.StopReasonToolUse:
		return "tool_use"
	case brtypes.StopReasonMaxTokens:
		return "max_tokens"
	case brtypes.StopReasonStopSequence:
		return "stop_sequence"
	default:
		return "end_turn"
	}
}

type bedrockEventStream interface {
	Events() <-chan brtypes.ConverseStreamOutput
	Err() error
}

func (s *bedrockServer) collectStream(stream bedrockEventStream, model string, emit func(anthropicEvent)) error {
	state := newBedrockStreamState(model)
	for event := range stream.Events() {
		for _, translated := range state.handle(event) {
			emit(translated)
		}
	}
	if err := stream.Err(); err != nil {
		return err
	}
	for _, translated := range state.finalize() {
		emit(translated)
	}
	return nil
}

func (s *bedrockServer) streamResponse(w http.ResponseWriter, stream bedrockEventStream, model string) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	err := s.collectStream(stream, model, func(ev anthropicEvent) {
		writeSSE(w, ev)
		if flusher != nil {
			flusher.Flush()
		}
	})
	if err != nil {
		writeSSE(w, errorEvent(bedrockErrorMessage(err)))
		if flusher != nil {
			flusher.Flush()
		}
	}
}

func bedrockErrorMessage(err error) string {
	_, _, message := bedrockError(err)
	return message
}

func (s *bedrockServer) unaryResponse(w http.ResponseWriter, stream bedrockEventStream, model string) {
	var events []anthropicEvent
	if err := s.collectStream(stream, model, func(ev anthropicEvent) { events = append(events, ev) }); err != nil {
		status, errType, message := bedrockError(err)
		writeAnthropicError(w, status, errType, message, false)
		return
	}
	msg, err := aggregateEvents(events, model)
	if err != nil {
		writeAnthropicError(w, http.StatusBadGateway, "api_error", err.Error(), false)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(msg)
}

func runBedrockServer(port int) {
	client, err := loadBedrockClient(context.Background())
	if err != nil {
		log.Fatalf("claude2openai bedrock: %v", err)
	}
	addr := net.JoinHostPort("127.0.0.1", strconv.Itoa(port))
	server := &http.Server{Addr: addr, Handler: (&bedrockServer{client: client}).mux(), ReadHeaderTimeout: 30 * time.Second}
	log.Printf("claude2openai bedrock: listening on http://%s, model=%s", addr, mapBedrockModel("claude-opus"))
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("claude2openai bedrock: %v", err)
	}
}

func runBedrockClaude(args []string) {
	client, err := loadBedrockClient(context.Background())
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai bedrock: %v\n", err)
		os.Exit(1)
	}
	proxy := httptest.NewServer((&bedrockServer{client: client}).mux())
	defer proxy.Close()

	claudePath, err := findClaude()
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai bedrock: %v\n", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "claude2openai bedrock: proxy started at %s, model=%s\n", proxy.URL, mapBedrockModel("claude-opus"))
	cmd := exec.Command(claudePath, args...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.Env = append(os.Environ(),
		"ANTHROPIC_BASE_URL="+proxy.URL,
		"ANTHROPIC_API_KEY=claude2bedrock",
		"ANTHROPIC_AUTH_TOKEN=claude2bedrock",
		"CLAUDE_CODE_USE_BEDROCK=",
	)
	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "claude2openai bedrock: failed to launch claude: %v\n", err)
		os.Exit(1)
	}
}

func runBedrockTest() {
	client, err := loadBedrockClient(context.Background())
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai bedrock test: %v\n", err)
		os.Exit(1)
	}
	proxy := httptest.NewServer((&bedrockServer{client: client}).mux())
	defer proxy.Close()

	model := mapBedrockModel("claude-test")
	testRequest := &anthropicRequest{
		Model: "claude-test", MaxTokens: 512,
		System:   jsonString("You are a concise assistant."),
		Messages: []anthropicMessage{{Role: "user", Content: jsonString("Reply with exactly: BEDROCK OPENAI OK")}},
	}
	effort := resolveBedrockEffort(testRequest, model)
	body, _ := json.Marshal(testRequest)
	request, err := http.NewRequest(http.MethodPost, proxy.URL+"/v1/messages", bytes.NewReader(body))
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai bedrock test: %v\n", err)
		os.Exit(1)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 120 * time.Second}).Do(request)
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai bedrock test: %v\n", err)
		os.Exit(1)
	}
	defer response.Body.Close()
	responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if response.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "claude2openai bedrock test: backend returned %s: %s\n", response.Status, truncate(string(responseBody), 1000))
		os.Exit(1)
	}
	var msg anthropicMessageOut
	if err := json.Unmarshal(responseBody, &msg); err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai bedrock test: invalid proxy response: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("model: %s\n", model)
	fmt.Printf("reasoning effort: %s\n", effort)
	var reply string
	for _, block := range msg.Content {
		if block["type"] == "text" {
			reply += fmt.Sprint(block["text"])
		}
	}
	if reply == "" {
		fmt.Fprintln(os.Stderr, "claude2openai bedrock test: response contained no text")
		os.Exit(1)
	}
	fmt.Printf("reply: %s\n", reply)
	fmt.Printf("usage: %d input / %d output tokens\n", msg.Usage["input_tokens"], msg.Usage["output_tokens"])
	fmt.Println("test: OK")
}
