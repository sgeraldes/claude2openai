package main

import (
	"encoding/json"
	"fmt"
)

// stream.go translates the upstream Responses API SSE event stream into
// Anthropic Messages API SSE events. streamTranslator is a pure state machine:
// feed it decoded upstream events with Handle, it returns the Anthropic events
// to emit. No I/O happens here, so the whole machine is unit-testable.

// ---------- upstream (Responses API) event ----------

type responsesEvent struct {
	Type         string          `json:"type"`
	OutputIndex  int             `json:"output_index"`
	ContentIndex int             `json:"content_index"`
	ItemID       string          `json:"item_id"`
	Delta        string          `json:"delta"`
	Text         string          `json:"text"`
	Arguments    string          `json:"arguments"`
	Item         *responsesOutputItem `json:"item"`
	Part         *struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"part"`
	Response  *responsesObject `json:"response"`
	Code      string           `json:"code"`
	Message   string           `json:"message"`
	Param     string           `json:"param"`
}

type responsesOutputItem struct {
	ID        string `json:"id"`
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Status    string `json:"status"`
}

type responsesObject struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Model  string `json:"model"`
	Error  *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Usage *struct {
		InputTokens         int `json:"input_tokens"`
		OutputTokens        int `json:"output_tokens"`
		TotalTokens         int `json:"total_tokens"`
		OutputTokensDetails *struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
}

// ---------- Anthropic event construction helpers ----------

type anthropicEvent struct {
	Event string // SSE event: line
	Data  any    // marshalled to the data: line
}

func messageStartEvent(msgID, model string) anthropicEvent {
	return anthropicEvent{"message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            msgID,
			"type":          "message",
			"role":          "assistant",
			"model":         model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage":         map[string]any{"input_tokens": 0, "output_tokens": 0},
		},
	}}
}

func blockStartEvent(index int, block map[string]any) anthropicEvent {
	return anthropicEvent{"content_block_start", map[string]any{
		"type": "content_block_start", "index": index, "content_block": block,
	}}
}

func blockDeltaEvent(index int, delta map[string]any) anthropicEvent {
	return anthropicEvent{"content_block_delta", map[string]any{
		"type": "content_block_delta", "index": index, "delta": delta,
	}}
}

func blockStopEvent(index int) anthropicEvent {
	return anthropicEvent{"content_block_stop", map[string]any{
		"type": "content_block_stop", "index": index,
	}}
}

func errorEvent(message string) anthropicEvent {
	return anthropicEvent{"error", map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "api_error", "message": message},
	}}
}

// ---------- the translator ----------

type blockKind int

const (
	blockNone blockKind = iota
	blockText
	blockToolUse
	blockThinking
)

type openBlock struct {
	kind  blockKind
	index int // anthropic content block index
	open  bool
}

type streamTranslator struct {
	clientModel   string // model name to echo back to the client
	wantThinking  bool   // client requested Anthropic thinking blocks
	messageID     string
	started       bool
	nextIndex     int
	blocks        map[int]*openBlock // keyed by upstream output_index
	sawToolUse    bool
	sawText       bool
	finishUsageIn  int
	finishUsageOut int
	stopReason    string
	finished      bool
}

func newStreamTranslator(clientModel string, wantThinking bool) *streamTranslator {
	return &streamTranslator{
		clientModel:  clientModel,
		wantThinking: wantThinking,
		messageID:    "msg_" + randomHex(12),
		stopReason:   "end_turn",
		blocks:       map[int]*openBlock{},
	}
}

func (t *streamTranslator) ensureStarted() []anthropicEvent {
	if t.started {
		return nil
	}
	t.started = true
	return []anthropicEvent{messageStartEvent(t.messageID, t.clientModel)}
}

func (t *streamTranslator) getBlock(outputIndex int) *openBlock {
	b, ok := t.blocks[outputIndex]
	if !ok {
		b = &openBlock{}
		t.blocks[outputIndex] = b
	}
	return b
}

func (t *streamTranslator) openTextBlock(outputIndex int) []anthropicEvent {
	var out []anthropicEvent
	b := t.getBlock(outputIndex)
	if b.open {
		return out
	}
	b.kind = blockText
	b.index = t.nextIndex
	t.nextIndex++
	b.open = true
	return append(out, blockStartEvent(b.index, map[string]any{"type": "text", "text": ""}))
}

func (t *streamTranslator) openThinkingBlock(outputIndex int) []anthropicEvent {
	var out []anthropicEvent
	b := t.getBlock(outputIndex)
	if b.open {
		return out
	}
	b.kind = blockThinking
	b.index = t.nextIndex
	t.nextIndex++
	b.open = true
	return append(out, blockStartEvent(b.index, map[string]any{"type": "thinking", "thinking": ""}))
}

func (t *streamTranslator) closeBlock(outputIndex int) []anthropicEvent {
	b, ok := t.blocks[outputIndex]
	if !ok || !b.open {
		return nil
	}
	b.open = false
	return []anthropicEvent{blockStopEvent(b.index)}
}

func (t *streamTranslator) closeAllBlocks() []anthropicEvent {
	var out []anthropicEvent
	// Close in ascending anthropic block order for a tidy stream.
	order := make([]int, 0, len(t.blocks))
	for _, b := range t.blocks {
		if b.open {
			order = append(order, b.index)
		}
	}
	for i := 0; i < len(order); i++ {
		for j := i + 1; j < len(order); j++ {
			if order[j] < order[i] {
				order[i], order[j] = order[j], order[i]
			}
		}
	}
	byIndex := map[int]int{} // anthropic index -> output_index
	for oi, b := range t.blocks {
		byIndex[b.index] = oi
	}
	for _, idx := range order {
		out = append(out, t.closeBlock(byIndex[idx])...)
	}
	return out
}

// Handle processes one upstream event and returns the Anthropic events to emit.
func (t *streamTranslator) Handle(ev *responsesEvent) []anthropicEvent {
	var out []anthropicEvent

	emit := func(evs ...anthropicEvent) {
		out = append(out, evs...)
	}

	// Everything after message_start; the first interesting event starts it.
	switch ev.Type {
	case "response.created":
		emit(t.ensureStarted()...)

	case "response.output_item.added":
		emit(t.ensureStarted()...)
		if ev.Item == nil {
			break
		}
		switch ev.Item.Type {
		case "function_call":
			b := t.getBlock(ev.OutputIndex)
			if !b.open {
				b.kind = blockToolUse
				b.index = t.nextIndex
				t.nextIndex++
				b.open = true
				t.sawToolUse = true
				callID := ev.Item.CallID
				if callID == "" {
					callID = "call_" + randomHex(12)
				}
				emit(blockStartEvent(b.index, map[string]any{
					"type": "tool_use", "id": callID, "name": ev.Item.Name, "input": map[string]any{},
				}))
			}
		}

	case "response.content_part.added":
		emit(t.ensureStarted()...)
		if ev.Part != nil && ev.Part.Type == "output_text" {
			emit(t.openTextBlock(ev.OutputIndex)...)
			if ev.Part.Text != "" {
				b := t.getBlock(ev.OutputIndex)
				t.sawText = true
				emit(blockDeltaEvent(b.index, map[string]any{"type": "text_delta", "text": ev.Part.Text}))
			}
		}

	case "response.output_text.delta":
		emit(t.ensureStarted()...)
		emit(t.openTextBlock(ev.OutputIndex)...)
		b := t.getBlock(ev.OutputIndex)
		t.sawText = true
		emit(blockDeltaEvent(b.index, map[string]any{"type": "text_delta", "text": ev.Delta}))

	case "response.output_text.done", "response.content_part.done":
		// Blocks close on output_item.done; some backends omit item.done for
		// message items, so also close here when we know the part is finished.
		if ev.Type == "response.content_part.done" {
			emit(t.closeBlock(ev.OutputIndex)...)
		}

	case "response.reasoning_summary_part.added":
		if !t.wantThinking {
			break
		}
		emit(t.ensureStarted()...)
		b := t.getBlock(ev.OutputIndex)
		wasOpen := b.open
		emit(t.openThinkingBlock(ev.OutputIndex)...)
		if wasOpen {
			// A second summary part in the same reasoning item: separate parts.
			emit(blockDeltaEvent(b.index, map[string]any{"type": "thinking_delta", "thinking": "\n\n"}))
		}

	case "response.reasoning_summary_text.delta":
		if !t.wantThinking {
			break
		}
		emit(t.ensureStarted()...)
		emit(t.openThinkingBlock(ev.OutputIndex)...)
		b := t.getBlock(ev.OutputIndex)
		emit(blockDeltaEvent(b.index, map[string]any{"type": "thinking_delta", "thinking": ev.Delta}))

	case "response.function_call_arguments.delta":
		emit(t.ensureStarted()...)
		b := t.getBlock(ev.OutputIndex)
		if b.open && b.kind == blockToolUse {
			emit(blockDeltaEvent(b.index, map[string]any{"type": "input_json_delta", "partial_json": ev.Delta}))
		}

	case "response.function_call_arguments.done":
		// Closed on output_item.done.
		emit(t.ensureStarted()...)

	case "response.output_item.done":
		emit(t.closeBlock(ev.OutputIndex)...)

	case "response.completed", "response.incomplete":
		emit(t.ensureStarted()...)
		emit(t.closeAllBlocks()...)
		t.finished = true
		t.stopReason = t.computeStopReason(ev)
		if ev.Response != nil && ev.Response.Usage != nil {
			t.finishUsageIn = ev.Response.Usage.InputTokens
			t.finishUsageOut = ev.Response.Usage.OutputTokens
		}
		emit(t.finishEvents()...)

	case "response.failed":
		emit(t.ensureStarted()...)
		emit(t.closeAllBlocks()...)
		t.finished = true
		msg := "upstream response failed"
		if ev.Response != nil && ev.Response.Error != nil && ev.Response.Error.Message != "" {
			msg = ev.Response.Error.Message
		}
		emit(errorEvent(msg))

	case "error":
		emit(t.ensureStarted()...)
		t.finished = true
		msg := ev.Message
		if msg == "" {
			msg = "upstream stream error"
		}
		emit(errorEvent(msg))

	default:
		// Unknown/uninteresting events (response.in_progress, rate_limits, etc.)
		// are ignored by design.
	}

	return out
}

func (t *streamTranslator) computeStopReason(ev *responsesEvent) string {
	if ev.Type == "response.incomplete" {
		if ev.Response != nil && ev.Response.IncompleteDetails != nil &&
			ev.Response.IncompleteDetails.Reason == "max_output_tokens" {
			return "max_tokens"
		}
	}
	if t.sawToolUse {
		return "tool_use"
	}
	return "end_turn"
}

func (t *streamTranslator) finishEvents() []anthropicEvent {
	return []anthropicEvent{
		{"message_delta", map[string]any{
			"type":  "message_delta",
			"delta": map[string]any{"stop_reason": t.stopReason, "stop_sequence": nil},
			"usage": map[string]any{"input_tokens": t.finishUsageIn, "output_tokens": t.finishUsageOut},
		}},
		{"message_stop", map[string]any{"type": "message_stop"}},
	}
}

// Finalize is called when the upstream stream ends. If the stream ended
// abruptly (no completed/failed event) it closes the Anthropic message cleanly
// so the client never hangs waiting for message_stop.
func (t *streamTranslator) Finalize() []anthropicEvent {
	if t.finished {
		return nil
	}
	t.finished = true
	var out []anthropicEvent
	out = append(out, t.ensureStarted()...)
	out = append(out, t.closeAllBlocks()...)
	if !t.sawToolUse && t.stopReason == "end_turn" {
		// keep end_turn
	} else if t.sawToolUse {
		t.stopReason = "tool_use"
	}
	out = append(out, t.finishEvents()...)
	return out
}

// ---------- aggregation for non-streaming clients ----------

// anthropicMessageOut is the non-streaming Messages API response body.
type anthropicMessageOut struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Role         string          `json:"role"`
	Model        string          `json:"model"`
	Content      []map[string]any `json:"content"`
	StopReason   string          `json:"stop_reason"`
	StopSequence any             `json:"stop_sequence"`
	Usage        map[string]int  `json:"usage"`
}

// aggregateEvents folds the emitted Anthropic events into a non-streaming
// Messages API response. Returns an error if the stream carried an error event.
func aggregateEvents(events []anthropicEvent, clientModel string) (*anthropicMessageOut, error) {
	out := &anthropicMessageOut{
		ID:      "msg_" + randomHex(12),
		Type:    "message",
		Role:    "assistant",
		Model:   clientModel,
		Content: []map[string]any{},
		Usage:   map[string]int{"input_tokens": 0, "output_tokens": 0},
	}
	blocks := map[int]map[string]any{}
	jsonArgs := map[int]string{}
	thinkingText := map[int]string{}
	textContent := map[int]string{}

	for _, ev := range events {
		data, _ := json.Marshal(ev.Data)
		var m map[string]any
		_ = json.Unmarshal(data, &m)
		switch ev.Event {
		case "message_start":
			if msg, ok := m["message"].(map[string]any); ok {
				if id, ok := msg["id"].(string); ok {
					out.ID = id
				}
			}
		case "content_block_start":
			idx := int(m["index"].(float64))
			block := m["content_block"].(map[string]any)
			blocks[idx] = block
		case "content_block_delta":
			idx := int(m["index"].(float64))
			delta := m["delta"].(map[string]any)
			switch delta["type"] {
			case "text_delta":
				textContent[idx] += delta["text"].(string)
			case "input_json_delta":
				jsonArgs[idx] += delta["partial_json"].(string)
			case "thinking_delta":
				thinkingText[idx] += delta["thinking"].(string)
			}
		case "message_delta":
			if delta, ok := m["delta"].(map[string]any); ok {
				if sr, ok := delta["stop_reason"].(string); ok {
					out.StopReason = sr
				}
			}
			if usage, ok := m["usage"].(map[string]any); ok {
				if v, ok := usage["input_tokens"].(float64); ok {
					out.Usage["input_tokens"] = int(v)
				}
				if v, ok := usage["output_tokens"].(float64); ok {
					out.Usage["output_tokens"] = int(v)
				}
			}
		case "error":
			if e, ok := m["error"].(map[string]any); ok {
				if msg, ok := e["message"].(string); ok {
					return nil, fmt.Errorf("%s", msg)
				}
			}
			return nil, fmt.Errorf("upstream error")
		}
	}

	// Emit blocks in index order with accumulated deltas applied.
	for i := 0; i < len(blocks); i++ {
		block, ok := blocks[i]
		if !ok {
			continue
		}
		switch block["type"] {
		case "text":
			block["text"] = textContent[i]
		case "tool_use":
			args := jsonArgs[i]
			var parsed any
			if err := json.Unmarshal([]byte(args), &parsed); err != nil {
				parsed = map[string]any{}
			}
			block["input"] = parsed
		case "thinking":
			block["thinking"] = thinkingText[i]
		}
		out.Content = append(out.Content, block)
	}
	if out.StopReason == "" {
		out.StopReason = "end_turn"
	}
	return out, nil
}
