package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
)

// GPT-5.6 models on Bedrock sometimes end a turn at effort "max" with only a
// redacted reasoning block and no text or tool call (observed on terra,
// 14-sep-2026: stop_reason end_turn, 22 output tokens, zero content blocks).
// Claude Code treats that as an empty assistant turn and stalls. When a stream
// closes without any content block, the proxy replays the request once with the
// next lower effort before surfacing anything to the client.
func debugEnabled() bool { return os.Getenv("BEDROCK_DEBUG") != "" }

func lowerEffort(effort string) (string, bool) {
	switch effort {
	case "max":
		return "high", true
	case "high":
		return "medium", true
	default:
		return "", false
	}
}

func withEffort(input *bedrockruntime.ConverseStreamInput, effort string) *bedrockruntime.ConverseStreamInput {
	clone := *input
	clone.AdditionalModelRequestFields = document.NewLazyDocument(map[string]any{"reasoning": map[string]string{"effort": effort}})
	return &clone
}

// bufferedRun collects the whole translated stream for one Converse call and
// reports whether it produced any content block.
type bufferedRun struct {
	events     []anthropicEvent
	hasContent bool
	err        error
}

func (s *bedrockServer) runOnce(ctx context.Context, input *bedrockruntime.ConverseStreamInput) bufferedRun {
	var run bufferedRun
	response, err := s.converseWithRetry(ctx, input)
	if err != nil {
		run.err = err
		return run
	}
	defer response.GetStream().Close()
	run.err = s.collectStream(response.GetStream(), aws.ToString(input.ModelId), func(ev anthropicEvent) {
		if debugEnabled() {
			log.Printf("bedrock messages: event %s %v", ev.Event, ev.Data)
		}
		if ev.Event == "content_block_start" {
			run.hasContent = true
		}
		run.events = append(run.events, ev)
	})
	return run
}

// runWithEmptyRetry executes the request and, if the model answered with no
// content block, retries once at a lower effort. It returns the events of the
// run that should reach the client.
func (s *bedrockServer) runWithEmptyRetry(ctx context.Context, input *bedrockruntime.ConverseStreamInput, effort string) bufferedRun {
	run := s.runOnce(ctx, input)
	if debugEnabled() {
		log.Printf("bedrock messages: run events=%d hasContent=%v err=%v", len(run.events), run.hasContent, run.err)
	}
	if run.err != nil || run.hasContent {
		return run
	}
	lower, ok := lowerEffort(effort)
	if !ok {
		return run
	}
	log.Printf("bedrock messages: empty reply from %q at effort=%q (only reasoning); retrying once at effort=%q", aws.ToString(input.ModelId), effort, lower)
	retry := s.runOnce(ctx, withEffort(input, lower))
	if retry.err != nil {
		return run
	}
	return retry
}

func (s *bedrockServer) writeRun(w http.ResponseWriter, run bufferedRun, stream bool, model string) {
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		for _, ev := range run.events {
			writeSSE(w, ev)
		}
		if run.err != nil {
			writeSSE(w, errorEvent(bedrockErrorMessage(run.err)))
		}
		if flusher != nil {
			flusher.Flush()
		}
		return
	}
	if run.err != nil {
		status, errType, message := bedrockError(run.err)
		writeAnthropicError(w, status, errType, message, false)
		return
	}
	msg, err := aggregateEvents(run.events, model)
	if err != nil {
		writeAnthropicError(w, http.StatusBadGateway, "api_error", err.Error(), false)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(msg)
}
