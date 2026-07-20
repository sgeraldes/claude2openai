package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

// translate.go converts an Anthropic Messages API request into an OpenAI
// Responses API request for the ChatGPT Codex backend. All functions here are
// pure (no I/O) so they are directly unit-testable.

// ---------- Anthropic request types ----------

type anthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	Messages    []anthropicMessage `json:"messages"`
	System      json.RawMessage    `json:"system,omitempty"` // string or []block
	Tools       []anthropicTool    `json:"tools,omitempty"`
	ToolChoice  json.RawMessage    `json:"tool_choice,omitempty"`
	Stream      bool               `json:"stream"`
	Temperature *float64           `json:"temperature,omitempty"`
	TopP        *float64           `json:"top_p,omitempty"`
	TopK        *int               `json:"top_k,omitempty"`
	Thinking    *anthropicThinking `json:"thinking,omitempty"`
	Metadata    map[string]any     `json:"metadata,omitempty"`
}

type anthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"` // string or []block
}

type anthropicBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"` // tool_result content
	IsError   bool            `json:"is_error,omitempty"`
	Source    *struct {
		Type      string `json:"type"`
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
	} `json:"source,omitempty"`
	Thinking string `json:"thinking,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// ---------- Responses request types ----------

type responsesRequest struct {
	Model             string           `json:"model"`
	Instructions      string           `json:"instructions,omitempty"`
	Input             []responsesItem  `json:"input"`
	Tools             []responsesTool  `json:"tools"`
	ToolChoice        any              `json:"tool_choice"`
	ParallelToolCalls bool             `json:"parallel_tool_calls"`
	Reasoning         *responsesReason `json:"reasoning,omitempty"`
	Store             bool             `json:"store"`
	Stream            bool             `json:"stream"`
	Include           []string         `json:"include"`
}

// Note: Anthropic's max_tokens is intentionally NOT forwarded. The ChatGPT
// codex backend rejects max_output_tokens with a 400 ("Unsupported parameter");
// output length is governed server-side per model.

type responsesReason struct {
	Effort  string `json:"effort"`
	Summary string `json:"summary"`
}

type responsesItem struct {
	Type      string                `json:"type"`
	Role      string                `json:"role,omitempty"`
	Content   []responsesContent    `json:"content,omitempty"`
	CallID    string                `json:"call_id,omitempty"`
	Name      string                `json:"name,omitempty"`
	Arguments string                `json:"arguments,omitempty"`
	Output    string                `json:"output,omitempty"`
}

type responsesContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type responsesTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      bool            `json:"strict"`
}

// ---------- model mapping ----------

// Codex model tiers. Verified against the account's live catalog
// (GET /backend-api/codex/models) on 2026-07-19 — note the backend's exact
// spelling is "gpt-5.6-terra" (not "tierra").
const (
	codexModelMain   = "gpt-5.6-sol"   // main thread / opus / fable tier, and fallback
	codexModelSonnet = "gpt-5.6-terra" // sonnet + subagent tier
	codexModelHaiku  = "gpt-5.6-luna"  // haiku tier
)

// codexDefaultModel is the fallback Codex model (used in usage/log text).
const codexDefaultModel = codexModelMain

// knownCodexModels is the live catalog verified against
// GET /backend-api/codex/models on 2026-07-19. Prefix rules in mapModel cover
// future ids; this set mainly documents what the account serves.
var knownCodexModels = map[string]bool{
	"gpt-5.6-sol":        true,
	"gpt-5.6-terra":      true,
	"gpt-5.6-luna":       true,
	"gpt-5.5":            true,
	"gpt-5.4":            true,
	"gpt-5.4-mini":       true,
	"gpt-5.3-codex-spark": true,
	"codex-auto-review":  true,
}

// mapModel routes an incoming model id to a Codex model. Recognized gpt/codex
// ids pass through unchanged. Claude ids map by tier:
//
//	opus / fable -> gpt-5.6-sol
//	sonnet       -> gpt-5.6-terra
//	haiku        -> gpt-5.6-luna
//
// Anything else falls back to the main tier. The CODEX_MODEL env var overrides
// the main/fallback tier only.
func mapModel(requested string) string {
	m := strings.ToLower(strings.TrimSpace(requested))
	if knownCodexModels[m] || strings.HasPrefix(m, "gpt-") || strings.Contains(m, "codex") {
		return requested
	}
	switch {
	case strings.Contains(m, "haiku"):
		return codexModelHaiku
	case strings.Contains(m, "sonnet"):
		return codexModelSonnet
	default: // opus, fable, and anything unrecognized
		return mainModel()
	}
}

// mainModel returns the main-tier/fallback Codex model (CODEX_MODEL overrides).
func mainModel() string {
	if v := strings.TrimSpace(os.Getenv("CODEX_MODEL")); v != "" {
		return v
	}
	return codexModelMain
}

// ---------- content translation ----------

// parseBlocks normalizes Anthropic content (string or block array) to blocks.
func parseBlocks(raw json.RawMessage) []anthropicBlock {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		if s == "" {
			return nil
		}
		return []anthropicBlock{{Type: "text", Text: s}}
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil
	}
	return blocks
}

// toolResultText flattens a tool_result's content (string or blocks) to text.
func toolResultText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return ""
	}
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" && b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// sanitizeToolName makes a name valid for the Responses function-calling API
// (^[a-zA-Z0-9_-]+$). Claude Code tool names are already compliant.
func sanitizeToolName(name string) string {
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	out := b.String()
	if out == "" {
		out = "tool"
	}
	if len(out) > 64 {
		out = out[:64]
	}
	return out
}

// translateSystem converts the Anthropic system field to Responses instructions.
func translateSystem(raw json.RawMessage) string {
	blocks := parseBlocks(raw)
	var parts []string
	for _, b := range blocks {
		if b.Type == "text" && strings.TrimSpace(b.Text) != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n\n")
}

// translateMessages converts Anthropic messages to Responses input items.
//
// Mapping:
//   - user/assistant text        -> message items (input_text / output_text)
//   - user image (base64)        -> input_image with a data URL
//   - assistant tool_use         -> function_call items
//   - user tool_result           -> function_call_output items
//   - thinking/redacted blocks   -> dropped (never replayed upstream)
func translateMessages(messages []anthropicMessage) []responsesItem {
	var items []responsesItem

	flushMessage := func(role string, content []responsesContent) {
		if len(content) == 0 {
			return
		}
		items = append(items, responsesItem{Type: "message", Role: role, Content: content})
	}

	for _, m := range messages {
		blocks := parseBlocks(m.Content)
		switch m.Role {
		case "user":
			var content []responsesContent
			for _, b := range blocks {
				switch b.Type {
				case "text":
					content = append(content, responsesContent{Type: "input_text", Text: b.Text})
				case "image":
					if b.Source != nil && b.Source.Type == "base64" && b.Source.Data != "" {
						dataURL := fmt.Sprintf("data:%s;base64,%s", b.Source.MediaType, b.Source.Data)
						content = append(content, responsesContent{Type: "input_image", ImageURL: dataURL})
					}
				case "tool_result":
					// Tool results are their own Responses items, so flush any
					// pending user text first.
					flushMessage("user", content)
					content = nil
					out := toolResultText(b.Content)
					if b.IsError {
						out = "Error: " + out
					}
					items = append(items, responsesItem{
						Type:   "function_call_output",
						CallID: b.ToolUseID,
						Output: out,
					})
				}
			}
			flushMessage("user", content)
		case "assistant":
			var content []responsesContent
			for _, b := range blocks {
				switch b.Type {
				case "text":
					content = append(content, responsesContent{Type: "output_text", Text: b.Text})
				case "tool_use":
					flushMessage("assistant", content)
					content = nil
					args := "{}"
					if len(b.Input) > 0 && string(b.Input) != "null" {
						args = string(b.Input)
					}
					items = append(items, responsesItem{
						Type:      "function_call",
						CallID:    b.ID,
						Name:      sanitizeToolName(b.Name),
						Arguments: args,
					})
				}
			}
			flushMessage("assistant", content)
		}
	}
	return items
}

// translateTools converts Anthropic tool definitions to Responses function tools.
func translateTools(tools []anthropicTool) []responsesTool {
	out := make([]responsesTool, 0, len(tools))
	for _, t := range tools {
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = json.RawMessage(`{"type":"object","properties":{}}`)
		}
		out = append(out, responsesTool{
			Type:        "function",
			Name:        sanitizeToolName(t.Name),
			Description: t.Description,
			Parameters:  schema,
			Strict:      false,
		})
	}
	return out
}

// translateToolChoice maps Anthropic tool_choice to the Responses form.
// Returns the value and whether parallel tool calls are allowed.
func translateToolChoice(raw json.RawMessage) (any, bool) {
	if len(raw) == 0 {
		return "auto", true
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		// Anthropic allows the bare string forms in some client versions.
		switch s {
		case "any":
			return "required", true
		case "none":
			return "none", false
		default:
			return "auto", true
		}
	}
	var tc struct {
		Type                  string `json:"type"`
		Name                  string `json:"name"`
		DisableParallelToolUse bool  `json:"disable_parallel_tool_use"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return "auto", true
	}
	parallel := !tc.DisableParallelToolUse
	switch tc.Type {
	case "any":
		return "required", parallel
	case "none":
		return "none", false
	case "tool":
		if tc.Name != "" {
			return map[string]string{"type": "function", "name": sanitizeToolName(tc.Name)}, false
		}
		return "required", parallel
	default: // "auto"
		return "auto", parallel
	}
}

// thinkingEffort maps an Anthropic thinking budget to a Codex reasoning effort.
func thinkingEffort(t *anthropicThinking) (effort string, ok bool) {
	if t == nil || t.Type != "enabled" {
		return "", false
	}
	switch {
	case t.BudgetTokens > 0 && t.BudgetTokens < 2048:
		return "low", true
	case t.BudgetTokens >= 15000:
		return "high", true
	default:
		return "medium", true
	}
}

// translateRequest builds the upstream Responses request. The upstream call is
// always streaming (the ChatGPT codex backend requires stream=true); the proxy
// aggregates when the client asked for a non-streaming response.
func translateRequest(req *anthropicRequest) *responsesRequest {
	toolChoice, parallel := translateToolChoice(req.ToolChoice)
	out := &responsesRequest{
		Model:             mapModel(req.Model),
		Instructions:      translateSystem(req.System),
		Input:             translateMessages(req.Messages),
		Tools:             translateTools(req.Tools),
		ToolChoice:        toolChoice,
		ParallelToolCalls: parallel,
		Store:             false,
		Stream:            true,
		Include:           []string{},
	}
	if out.Tools == nil {
		out.Tools = []responsesTool{}
	}
	if out.Input == nil {
		out.Input = []responsesItem{}
	}
	if effort, ok := thinkingEffort(req.Thinking); ok {
		out.Reasoning = &responsesReason{Effort: effort, Summary: "auto"}
	}
	return out
}
