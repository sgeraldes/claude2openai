package main

import (
	"encoding/json"
	"strings"
	"testing"
)

// feed pumps raw SSE text through parseResponsesSSE + the translator and
// returns all emitted Anthropic events.
func feed(t *testing.T, rawSSE string, clientModel string, wantThinking bool) []anthropicEvent {
	t.Helper()
	tr := newStreamTranslator(clientModel, wantThinking)
	var out []anthropicEvent
	for ev := range parseResponsesSSE(strings.NewReader(rawSSE)) {
		if ev.parseErr != nil {
			t.Fatalf("parse error: %v", ev.parseErr)
		}
		out = append(out, tr.Handle(ev.event)...)
	}
	out = append(out, tr.Finalize()...)
	return out
}

func eventTypes(events []anthropicEvent) []string {
	var out []string
	for _, e := range events {
		out = append(out, e.Event)
	}
	return out
}

func TestStreamTextResponse(t *testing.T) {
	raw := `event: response.created
data: {"type":"response.created","response":{"id":"resp_1","status":"in_progress"}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"m1","type":"message","status":"in_progress"}}

event: response.content_part.added
data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello"}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":" world"}

event: response.content_part.done
data: {"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hello world"}}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"m1","type":"message","status":"completed"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":12,"output_tokens":5,"total_tokens":17}}}

`
	events := feed(t, raw, "claude-sonnet-4-5", false)
	want := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	got := eventTypes(events)
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("event sequence:\n got %v\nwant %v", got, want)
	}

	// Verify message_start shape echoes the client model.
	ms, _ := json.Marshal(events[0].Data)
	if !strings.Contains(string(ms), `"model":"claude-sonnet-4-5"`) {
		t.Errorf("message_start should echo client model: %s", ms)
	}
	// Text deltas carry the text.
	d1, _ := json.Marshal(events[2].Data)
	if !strings.Contains(string(d1), `"text":"Hello"`) {
		t.Errorf("first delta: %s", d1)
	}
	// Final usage lands in message_delta.
	md, _ := json.Marshal(events[5].Data)
	if !strings.Contains(string(md), `"output_tokens":5`) || !strings.Contains(string(md), `"input_tokens":12`) {
		t.Errorf("message_delta usage: %s", md)
	}
	if !strings.Contains(string(md), `"stop_reason":"end_turn"`) {
		t.Errorf("stop_reason: %s", md)
	}

	// Aggregation round-trip.
	msg, err := aggregateEvents(events, "claude-sonnet-4-5")
	if err != nil {
		t.Fatalf("aggregate: %v", err)
	}
	if len(msg.Content) != 1 || msg.Content[0]["type"] != "text" || msg.Content[0]["text"] != "Hello world" {
		t.Errorf("aggregated content: %+v", msg.Content)
	}
	if msg.StopReason != "end_turn" || msg.Usage["output_tokens"] != 5 || msg.Usage["input_tokens"] != 12 {
		t.Errorf("aggregated meta: %+v %+v", msg.StopReason, msg.Usage)
	}
}

func TestStreamToolCall(t *testing.T) {
	raw := `event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc1","type":"function_call","call_id":"call_abc","name":"Bash","arguments":"","status":"in_progress"}}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\"command\":"}

event: response.function_call_arguments.delta
data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"\"ls\"}"}

event: response.function_call_arguments.done
data: {"type":"response.function_call_arguments.done","output_index":0,"arguments":"{\"command\":\"ls\"}"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc1","type":"function_call","call_id":"call_abc","name":"Bash","arguments":"{\"command\":\"ls\"}","status":"completed"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":20,"output_tokens":7,"total_tokens":27}}}

`
	events := feed(t, raw, "claude-sonnet-4-5", false)
	types := eventTypes(events)
	want := []string{
		"message_start",
		"content_block_start", // tool_use
		"content_block_delta", // input_json_delta
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}
	if strings.Join(types, ",") != strings.Join(want, ",") {
		t.Fatalf("event sequence:\n got %v\nwant %v", types, want)
	}

	bs, _ := json.Marshal(events[1].Data)
	if !strings.Contains(string(bs), `"type":"tool_use"`) || !strings.Contains(string(bs), `"id":"call_abc"`) || !strings.Contains(string(bs), `"name":"Bash"`) {
		t.Errorf("tool_use block start: %s", bs)
	}
	md, _ := json.Marshal(events[5].Data)
	if !strings.Contains(string(md), `"stop_reason":"tool_use"`) {
		t.Errorf("stop_reason should be tool_use: %s", md)
	}

	msg, err := aggregateEvents(events, "claude-sonnet-4-5")
	if err != nil {
		t.Fatalf("aggregate: %v", err)
	}
	if len(msg.Content) != 1 || msg.Content[0]["type"] != "tool_use" {
		t.Fatalf("aggregated: %+v", msg.Content)
	}
	input, ok := msg.Content[0]["input"].(map[string]any)
	if !ok || input["command"] != "ls" {
		t.Errorf("aggregated tool input: %+v", msg.Content[0]["input"])
	}
}

func TestStreamThinking(t *testing.T) {
	raw := `event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"r1","type":"reasoning"}}

event: response.reasoning_summary_part.added
data: {"type":"response.reasoning_summary_part.added","output_index":0,"summary_index":0,"part":{"type":"summary_text","text":""}}

event: response.reasoning_summary_text.delta
data: {"type":"response.reasoning_summary_text.delta","output_index":0,"summary_index":0,"delta":"thinking hard"}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"r1","type":"reasoning"}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":1,"content_index":0,"delta":"answer"}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","usage":{"input_tokens":3,"output_tokens":9,"total_tokens":12}}}

`
	// Thinking requested: reasoning maps to a thinking block.
	events := feed(t, raw, "claude-x", true)
	types := eventTypes(events)
	joined := strings.Join(types, ",")
	if !strings.Contains(joined, "content_block_start,content_block_delta") {
		t.Fatalf("thinking events: %v", types)
	}
	var sawThinkingDelta bool
	for _, e := range events {
		b, _ := json.Marshal(e.Data)
		if strings.Contains(string(b), "thinking_delta") && strings.Contains(string(b), "thinking hard") {
			sawThinkingDelta = true
		}
	}
	if !sawThinkingDelta {
		t.Errorf("expected a thinking_delta with the summary text: %v", types)
	}

	// Thinking not requested: reasoning events are dropped entirely.
	events2 := feed(t, raw, "claude-x", false)
	for _, e := range events2 {
		b, _ := json.Marshal(e.Data)
		if strings.Contains(string(b), "thinking") {
			t.Errorf("no thinking blocks expected when not requested: %s", b)
		}
	}
	msg, err := aggregateEvents(events2, "claude-x")
	if err != nil {
		t.Fatalf("aggregate: %v", err)
	}
	if len(msg.Content) != 1 || msg.Content[0]["type"] != "text" {
		t.Errorf("only the text block should remain: %+v", msg.Content)
	}
}

func TestStreamErrorEvent(t *testing.T) {
	raw := `event: error
data: {"type":"error","code":"rate_limit","message":"slow down"}

`
	events := feed(t, raw, "claude-x", false)
	var sawErr bool
	for _, e := range events {
		if e.Event == "error" {
			sawErr = true
			b, _ := json.Marshal(e.Data)
			if !strings.Contains(string(b), "slow down") {
				t.Errorf("error payload: %s", b)
			}
		}
	}
	if !sawErr {
		t.Fatalf("expected an error event, got %v", eventTypes(events))
	}
	if _, err := aggregateEvents(events, "claude-x"); err == nil || !strings.Contains(err.Error(), "slow down") {
		t.Errorf("aggregate should surface the upstream error, got %v", err)
	}
}

func TestStreamAbruptEnd(t *testing.T) {
	// Stream dies after a text delta with no completed event.
	raw := `event: response.output_text.delta
data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"partial"}

`
	events := feed(t, raw, "claude-x", false)
	types := eventTypes(events)
	// Finalize must still close block + emit message_delta/message_stop.
	if types[len(types)-1] != "message_stop" {
		t.Fatalf("abrupt stream must still terminate cleanly: %v", types)
	}
	if !strings.Contains(strings.Join(types, ","), "content_block_stop") {
		t.Errorf("open block must be closed on finalize: %v", types)
	}
}

func TestStreamIncompleteMaxTokens(t *testing.T) {
	raw := `event: response.incomplete
data: {"type":"response.incomplete","response":{"id":"r","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}

`
	events := feed(t, raw, "claude-x", false)
	md, _ := json.Marshal(events[len(events)-2].Data)
	if !strings.Contains(string(md), `"stop_reason":"max_tokens"`) {
		t.Errorf("incomplete/max_output_tokens should map to max_tokens: %s", md)
	}
}

func TestParseResponsesSSEDoneSkipped(t *testing.T) {
	raw := "data: [DONE]\n\n"
	n := 0
	for range parseResponsesSSE(strings.NewReader(raw)) {
		n++
	}
	if n != 0 {
		t.Errorf("[DONE] should be skipped, got %d events", n)
	}
}

func TestAggregateEmptyStream(t *testing.T) {
	msg, err := aggregateEvents(nil, "claude-x")
	if err != nil {
		t.Fatalf("empty stream should aggregate: %v", err)
	}
	if msg.StopReason != "end_turn" || len(msg.Content) != 0 {
		t.Errorf("empty aggregation: %+v", msg)
	}
}
