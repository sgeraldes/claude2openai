package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMapModel(t *testing.T) {
	t.Setenv("CODEX_MODEL", "")
	cases := []struct{ in, want string }{
		// Claude tiers
		{"claude-sonnet-4-5", codexModelSonnet},
		{"claude-3-5-sonnet-20241022", codexModelSonnet},
		{"claude-fable-5", codexModelMain},
		{"claude-opus-4-8-20260101", codexModelMain},
		{"claude-opus-4-5", codexModelMain},
		{"claude-haiku-4-5", codexModelHaiku},
		{"claude-3-5-haiku-20241022", codexModelHaiku},
		// gpt/codex ids pass through
		{"gpt-5.5", "gpt-5.5"},
		{"gpt-5.6-sol", "gpt-5.6-sol"},
		{"gpt-5.6-terra", "gpt-5.6-terra"},
		{"gpt-5.6-luna", "gpt-5.6-luna"},
		{"gpt-5.3-codex-spark", "gpt-5.3-codex-spark"},
		{"codex-auto-review", "codex-auto-review"},
		{"gpt-99-future", "gpt-99-future"}, // unknown gpt- ids pass through
		// fallback -> main tier
		{"", codexModelMain},
		{"llama3", codexModelMain},
	}
	for _, c := range cases {
		if got := mapModel(c.in); got != c.want {
			t.Errorf("mapModel(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestMapModelEnvOverride(t *testing.T) {
	t.Setenv("CODEX_MODEL", "gpt-5.4-mini")
	// Override applies to the main/fallback tier.
	if got := mapModel("claude-opus-4-5"); got != "gpt-5.4-mini" {
		t.Errorf("main tier with CODEX_MODEL override = %q, want gpt-5.4-mini", got)
	}
	if got := mapModel("something-unrecognized"); got != "gpt-5.4-mini" {
		t.Errorf("fallback with CODEX_MODEL override = %q, want gpt-5.4-mini", got)
	}
	// Sonnet/haiku tiers are NOT affected by the override.
	if got := mapModel("claude-sonnet-4-5"); got != codexModelSonnet {
		t.Errorf("sonnet tier should ignore override, got %q", got)
	}
	if got := mapModel("claude-haiku-4-5"); got != codexModelHaiku {
		t.Errorf("haiku tier should ignore override, got %q", got)
	}
	// Recognized ids still pass through under an override.
	if got := mapModel("gpt-5.5"); got != "gpt-5.5" {
		t.Errorf("recognized model should pass through, got %q", got)
	}
}

func TestTranslateSystemStringAndBlocks(t *testing.T) {
	if got := translateSystem(json.RawMessage(`"You are helpful."`)); got != "You are helpful." {
		t.Errorf("string system: got %q", got)
	}
	blocks := json.RawMessage(`[{"type":"text","text":"Part one."},{"type":"text","text":"Part two."}]`)
	if got := translateSystem(blocks); got != "Part one.\n\nPart two." {
		t.Errorf("block system: got %q", got)
	}
	if got := translateSystem(nil); got != "" {
		t.Errorf("nil system: got %q", got)
	}
}

func TestTranslateMessagesSimpleText(t *testing.T) {
	msgs := []anthropicMessage{
		{Role: "user", Content: json.RawMessage(`"hello"`)},
		{Role: "assistant", Content: json.RawMessage(`[{"type":"text","text":"hi there"}]`)},
	}
	items := translateMessages(msgs)
	if len(items) != 2 {
		t.Fatalf("expected 2 items, got %d: %+v", len(items), items)
	}
	if items[0].Type != "message" || items[0].Role != "user" || items[0].Content[0].Type != "input_text" || items[0].Content[0].Text != "hello" {
		t.Errorf("bad user item: %+v", items[0])
	}
	if items[1].Role != "assistant" || items[1].Content[0].Type != "output_text" || items[1].Content[0].Text != "hi there" {
		t.Errorf("bad assistant item: %+v", items[1])
	}
}

func TestTranslateMessagesToolUseAndResult(t *testing.T) {
	msgs := []anthropicMessage{
		{Role: "assistant", Content: json.RawMessage(`[
			{"type":"text","text":"Let me check."},
			{"type":"tool_use","id":"toolu_01ABC","name":"Bash","input":{"command":"ls"}}
		]`)},
		{Role: "user", Content: json.RawMessage(`[
			{"type":"tool_result","tool_use_id":"toolu_01ABC","content":[{"type":"text","text":"file1.txt\nfile2.txt"}]},
			{"type":"text","text":"what about go files?"}
		]`)},
	}
	items := translateMessages(msgs)
	if len(items) != 4 {
		t.Fatalf("expected 4 items, got %d: %+v", len(items), items)
	}
	// assistant text message
	if items[0].Type != "message" || items[0].Role != "assistant" {
		t.Errorf("item0: %+v", items[0])
	}
	// function_call
	fc := items[1]
	if fc.Type != "function_call" || fc.CallID != "toolu_01ABC" || fc.Name != "Bash" {
		t.Errorf("function_call: %+v", fc)
	}
	var args map[string]string
	if err := json.Unmarshal([]byte(fc.Arguments), &args); err != nil || args["command"] != "ls" {
		t.Errorf("function_call arguments: %q (%v)", fc.Arguments, err)
	}
	// function_call_output
	fco := items[2]
	if fco.Type != "function_call_output" || fco.CallID != "toolu_01ABC" || !strings.Contains(fco.Output, "file1.txt") {
		t.Errorf("function_call_output: %+v", fco)
	}
	// trailing user text
	if items[3].Type != "message" || items[3].Role != "user" || items[3].Content[0].Text != "what about go files?" {
		t.Errorf("trailing user message: %+v", items[3])
	}
}

func TestTranslateMessagesToolResultErrorAndString(t *testing.T) {
	msgs := []anthropicMessage{
		{Role: "user", Content: json.RawMessage(`[
			{"type":"tool_result","tool_use_id":"toolu_x","content":"boom","is_error":true}
		]`)},
	}
	items := translateMessages(msgs)
	if len(items) != 1 || items[0].Type != "function_call_output" {
		t.Fatalf("items: %+v", items)
	}
	if !strings.HasPrefix(items[0].Output, "Error: boom") {
		t.Errorf("error tool result should be prefixed, got %q", items[0].Output)
	}
}

// An empty tool result must still serialize an `output` key. The field is
// tagged omitempty on a struct shared with other item types, so without a
// placeholder the key vanishes and the backend rejects the whole conversation
// with 400 "Missing required parameter: input[N].output".
func TestTranslateMessagesEmptyToolResultKeepsOutputKey(t *testing.T) {
	cases := []struct {
		name    string
		content string
	}{
		{"empty string", `""`},
		{"empty block list", `[]`},
		{"blocks with no text", `[{"type":"text","text":""}]`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			msgs := []anthropicMessage{
				{Role: "user", Content: json.RawMessage(
					`[{"type":"tool_result","tool_use_id":"toolu_empty","content":` + tc.content + `}]`)},
			}
			items := translateMessages(msgs)
			if len(items) != 1 || items[0].Type != "function_call_output" {
				t.Fatalf("items: %+v", items)
			}
			if items[0].Output == "" {
				t.Fatalf("empty tool result produced an empty Output; omitempty will drop the key")
			}
			blob, err := json.Marshal(items[0])
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			var round map[string]any
			if err := json.Unmarshal(blob, &round); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if _, ok := round["output"]; !ok {
				t.Errorf("serialized function_call_output is missing the output key: %s", blob)
			}
		})
	}
}

func TestTranslateMessagesImage(t *testing.T) {
	msgs := []anthropicMessage{
		{Role: "user", Content: json.RawMessage(`[
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}},
			{"type":"text","text":"what is this?"}
		]`)},
	}
	items := translateMessages(msgs)
	if len(items) != 1 || len(items[0].Content) != 2 {
		t.Fatalf("items: %+v", items)
	}
	img := items[0].Content[0]
	if img.Type != "input_image" || img.ImageURL != "data:image/png;base64,aGVsbG8=" {
		t.Errorf("image content: %+v", img)
	}
}

func TestTranslateMessagesDropsThinking(t *testing.T) {
	msgs := []anthropicMessage{
		{Role: "assistant", Content: json.RawMessage(`[
			{"type":"thinking","thinking":"hmm","signature":"sig"},
			{"type":"text","text":"answer"}
		]`)},
	}
	items := translateMessages(msgs)
	if len(items) != 1 || items[0].Content[0].Text != "answer" {
		t.Errorf("thinking blocks must be dropped from history: %+v", items)
	}
}

func TestTranslateTools(t *testing.T) {
	tools := []anthropicTool{
		{Name: "Bash", Description: "run commands", InputSchema: json.RawMessage(`{"type":"object","properties":{"command":{"type":"string"}}}`)},
		{Name: "bad name!", InputSchema: nil},
	}
	out := translateTools(tools)
	if len(out) != 2 {
		t.Fatalf("tools: %+v", out)
	}
	if out[0].Type != "function" || out[0].Name != "Bash" || out[0].Strict {
		t.Errorf("tool0: %+v", out[0])
	}
	if string(out[0].Parameters) != `{"type":"object","properties":{"command":{"type":"string"}}}` {
		t.Errorf("schema not preserved: %s", out[0].Parameters)
	}
	if out[1].Name != "bad_name_" {
		t.Errorf("name not sanitized: %q", out[1].Name)
	}
	if string(out[1].Parameters) != `{"type":"object","properties":{}}` {
		t.Errorf("nil schema should get default object schema: %s", out[1].Parameters)
	}
}

func TestTranslateToolChoice(t *testing.T) {
	tc, parallel := translateToolChoice(nil)
	if tc != "auto" || !parallel {
		t.Errorf("nil: %v %v", tc, parallel)
	}
	tc, _ = translateToolChoice(json.RawMessage(`{"type":"any"}`))
	if tc != "required" {
		t.Errorf("any: %v", tc)
	}
	tc, parallel = translateToolChoice(json.RawMessage(`{"type":"auto","disable_parallel_tool_use":true}`))
	if tc != "auto" || parallel {
		t.Errorf("auto+disable_parallel: %v %v", tc, parallel)
	}
	tc, parallel = translateToolChoice(json.RawMessage(`{"type":"tool","name":"Bash"}`))
	m, ok := tc.(map[string]string)
	if !ok || m["type"] != "function" || m["name"] != "Bash" || parallel {
		t.Errorf("forced tool: %v %v", tc, parallel)
	}
}

func TestThinkingEffort(t *testing.T) {
	if _, ok := thinkingEffort(nil); ok {
		t.Error("nil thinking should be disabled")
	}
	if _, ok := thinkingEffort(&anthropicThinking{Type: "disabled"}); ok {
		t.Error("disabled thinking should be disabled")
	}
	if e, ok := thinkingEffort(&anthropicThinking{Type: "enabled", BudgetTokens: 1024}); !ok || e != "low" {
		t.Errorf("small budget: %v %v", e, ok)
	}
	if e, ok := thinkingEffort(&anthropicThinking{Type: "enabled", BudgetTokens: 8000}); !ok || e != "medium" {
		t.Errorf("medium budget: %v %v", e, ok)
	}
	if e, ok := thinkingEffort(&anthropicThinking{Type: "enabled", BudgetTokens: 20000}); !ok || e != "high" {
		t.Errorf("large budget: %v %v", e, ok)
	}
}

func TestTranslateRequestEndToEnd(t *testing.T) {
	t.Setenv("CODEX_MODEL", "")
	req := &anthropicRequest{
		Model:     "claude-opus-4-8",
		MaxTokens: 4096,
		System:    json.RawMessage(`"sys"`),
		Messages: []anthropicMessage{
			{Role: "user", Content: json.RawMessage(`"hi"`)},
		},
		Tools: []anthropicTool{
			{Name: "Read", InputSchema: json.RawMessage(`{"type":"object"}`)},
		},
		Thinking: &anthropicThinking{Type: "enabled", BudgetTokens: 5000},
	}
	out := translateRequest(req)
	if out.Model != codexModelMain {
		t.Errorf("model: %q", out.Model)
	}
	// Sonnet requests route to the sonnet tier.
	sonnetOut := translateRequest(&anthropicRequest{Model: "claude-sonnet-4-5", MaxTokens: 10})
	if sonnetOut.Model != codexModelSonnet {
		t.Errorf("sonnet tier: %q", sonnetOut.Model)
	}
	if !out.Stream || out.Store {
		t.Error("upstream must always be stream=true store=false")
	}
	if out.Instructions != "sys" {
		t.Errorf("instructions: %q", out.Instructions)
	}
	if out.Reasoning == nil || out.Reasoning.Effort != "medium" || out.Reasoning.Summary != "auto" {
		t.Errorf("reasoning: %+v", out.Reasoning)
	}
	if out.Include == nil || len(out.Include) != 0 {
		t.Errorf("include should be an empty array, got %v", out.Include)
	}
	if out.Tools == nil || len(out.Tools) != 1 {
		t.Errorf("tools: %+v", out.Tools)
	}
	// Must be valid JSON (the backend will reject otherwise).
	if _, err := json.Marshal(out); err != nil {
		t.Fatalf("marshal: %v", err)
	}
}

func TestTranslateRequestNoThinking(t *testing.T) {
	out := translateRequest(&anthropicRequest{Model: "gpt-5.5", MaxTokens: 10})
	if out.Reasoning != nil {
		t.Errorf("reasoning should be omitted without thinking: %+v", out.Reasoning)
	}
	if out.Model != "gpt-5.5" {
		t.Errorf("gpt model should pass through: %q", out.Model)
	}
}
