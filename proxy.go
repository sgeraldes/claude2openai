package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

// proxy.go implements the localhost HTTP server Claude Code talks to, plus the
// upstream client for the ChatGPT Codex backend. Contract verified live on
// 2026-07-19 against codex-cli 0.144.1's backend:
//
//	POST https://chatgpt.com/backend-api/codex/responses
//	headers: Authorization: Bearer <access_token>, chatgpt-account-id: <account_id>,
//	         originator: codex_cli_rs, OpenAI-Beta: responses=experimental
//	body:    Responses API, always stream=true, store=false
const (
	codexResponsesURL = "https://chatgpt.com/backend-api/codex/responses"
	codexOriginator   = "codex_cli_rs"
	codexUserAgent    = "codex_cli_rs/0.144.1 (Windows NT 10.0; x86_64)"
)

func randomHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

type proxyServer struct {
	client *http.Client
}

func newProxyServer() *proxyServer {
	return &proxyServer{
		// No overall timeout: model streams can run for many minutes. Per-request
		// cancellation comes from the incoming request's context.
		client: &http.Client{},
	}
}

func (s *proxyServer) mux() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("OK"))
	})
	mux.HandleFunc("POST /v1/messages", s.handleMessages)
	mux.HandleFunc("POST /v1/messages/count_tokens", s.handleCountTokens)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeAnthropicError(w, http.StatusNotFound, "not_found_error", "unknown endpoint: "+r.URL.Path, false)
	})
	return mux
}

// writeAnthropicError renders an error in the Anthropic Messages API shape.
// When stream is true the error rides inside a 200 SSE stream (the client's
// retry logic treats in-stream errors as terminal, matching Claude Code's
// expectations for proxied backends).
func writeAnthropicError(w http.ResponseWriter, status int, errType, message string, stream bool) {
	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)
		writeSSE(w, errorEvent(message))
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"type":  "error",
		"error": map[string]any{"type": errType, "message": message},
	})
}

func writeSSE(w io.Writer, ev anthropicEvent) {
	data, err := json.Marshal(ev.Data)
	if err != nil {
		return
	}
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.Event, data)
}

func (s *proxyServer) handleMessages(w http.ResponseWriter, r *http.Request) {
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
	if req.MaxTokens <= 0 {
		req.MaxTokens = 8192
	}

	upstream := translateRequest(&req)
	effort := "default"
	if upstream.Reasoning != nil {
		effort = upstream.Reasoning.Effort
	}
	log.Printf("messages: model %q -> %q, effort=%q, %d input items, %d tools, stream=%v",
		req.Model, upstream.Model, effort, len(upstream.Input), len(upstream.Tools), req.Stream)

	payload, err := json.Marshal(upstream)
	if err != nil {
		writeAnthropicError(w, http.StatusInternalServerError, "api_error", "cannot encode upstream request", req.Stream)
		return
	}

	resp, err := s.doUpstream(r.Context(), payload)
	if err != nil {
		writeAnthropicError(w, http.StatusBadGateway, "api_error", sanitizeUpstreamError(err), req.Stream)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		msg := sanitizeUpstreamError(fmt.Errorf("backend returned %s: %s", resp.Status, truncate(string(errBody), 500)))
		status := resp.StatusCode
		if status == http.StatusUnauthorized {
			msg += " — your Codex ChatGPT session is not accepted; run `codex login` again"
		}
		writeAnthropicError(w, status, "api_error", msg, req.Stream)
		return
	}

	if req.Stream {
		s.streamResponse(w, resp.Body, &req, upstream)
		return
	}
	s.unaryResponse(w, resp.Body, &req)
}

// doUpstream POSTs the Responses request with the ChatGPT OAuth headers. On a
// 401 it forces one token refresh and retries once.
func (s *proxyServer) doUpstream(ctx context.Context, payload []byte) (*http.Response, error) {
	send := func() (*http.Response, error) {
		token, accountID, err := ensureFreshToken()
		if err != nil {
			return nil, err
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, codexResponsesURL, bytes.NewReader(payload))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("chatgpt-account-id", accountID)
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")
		req.Header.Set("originator", codexOriginator)
		req.Header.Set("OpenAI-Beta", "responses=experimental")
		req.Header.Set("User-Agent", codexUserAgent)
		return s.client.Do(req)
	}

	resp, err := send()
	if err != nil {
		return nil, err
	}
	if resp.StatusCode == http.StatusUnauthorized {
		resp.Body.Close()
		// Force a refresh by expiring our cached view: re-read the file and, if
		// another writer already refreshed it, the new token is picked up;
		// otherwise the refresh flow runs inside ensureFreshToken.
		if rerr := forceRefresh(); rerr != nil {
			return nil, fmt.Errorf("backend rejected the token (401) and refresh failed: %w; run `codex login` again", rerr)
		}
		resp, err = send()
	}
	return resp, err
}

// streamResponse pipes the upstream SSE stream through the translator to the
// client as Anthropic SSE.
func (s *proxyServer) streamResponse(w http.ResponseWriter, body io.Reader, req *anthropicRequest, upstream *responsesRequest) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	flusher, _ := w.(http.Flusher)
	translator := newStreamTranslator(req.Model, wantsThinking(req))

	for ev := range parseResponsesSSE(body) {
		if ev.parseErr != nil {
			log.Printf("stream: upstream parse error: %v", ev.parseErr)
			continue
		}
		for _, out := range translator.Handle(ev.event) {
			writeSSE(w, out)
		}
		if flusher != nil {
			flusher.Flush()
		}
	}
	for _, out := range translator.Finalize() {
		writeSSE(w, out)
	}
	if flusher != nil {
		flusher.Flush()
	}
}

// unaryResponse consumes the upstream stream and emits one JSON Messages
// response (the backend itself only streams, so we aggregate).
func (s *proxyServer) unaryResponse(w http.ResponseWriter, body io.Reader, req *anthropicRequest) {
	translator := newStreamTranslator(req.Model, wantsThinking(req))
	var events []anthropicEvent
	for ev := range parseResponsesSSE(body) {
		if ev.parseErr != nil {
			continue
		}
		events = append(events, translator.Handle(ev.event)...)
	}
	events = append(events, translator.Finalize()...)

	msg, err := aggregateEvents(events, req.Model)
	if err != nil {
		writeAnthropicError(w, http.StatusBadGateway, "api_error", sanitizeUpstreamError(err), false)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(msg)
}

func wantsThinking(req *anthropicRequest) bool {
	_, ok := thinkingEffort(req.Thinking)
	return ok
}

// handleCountTokens answers a cheap local estimate; Claude Code uses this
// endpoint only for UI accounting.
func (s *proxyServer) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 64<<20))
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "cannot read request body", false)
		return
	}
	// ~4 chars per token across all textual content.
	n := len(body) / 4
	if n < 1 {
		n = 1
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"input_tokens": n})
}

// ---------- upstream SSE parsing ----------

type parsedEvent struct {
	event    *responsesEvent
	parseErr error
}

// parseResponsesSSE decodes the upstream text/event-stream body into events.
// The `data:` payload carries a `type` field; the SSE `event:` line mirrors it.
func parseResponsesSSE(body io.Reader) chan parsedEvent {
	ch := make(chan parsedEvent, 16)
	go func() {
		defer close(ch)
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 256*1024), 8*1024*1024)
		var dataLines []string
		flush := func() {
			if len(dataLines) == 0 {
				return
			}
			data := strings.Join(dataLines, "\n")
			dataLines = nil
			if data == "[DONE]" {
				return
			}
			var ev responsesEvent
			if err := json.Unmarshal([]byte(data), &ev); err != nil {
				ch <- parsedEvent{parseErr: fmt.Errorf("bad event data: %w", err)}
				return
			}
			ch <- parsedEvent{event: &ev}
		}
		for scanner.Scan() {
			line := scanner.Text()
			switch {
			case line == "":
				flush()
			case strings.HasPrefix(line, "data:"):
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			default:
				// event:/id:/retry:/comment lines are not needed: the payload's
				// type field is authoritative.
			}
		}
		flush()
		if err := scanner.Err(); err != nil {
			ch <- parsedEvent{parseErr: err}
		}
	}()
	return ch
}

// sanitizeUpstreamError strips anything that could resemble token material
// from an upstream error before it reaches logs or clients.
func sanitizeUpstreamError(err error) string {
	msg := err.Error()
	// Tokens are long; redact any very long whitespace-free run just in case.
	fields := strings.Fields(msg)
	for i, f := range fields {
		if len(f) > 80 {
			fields[i] = f[:12] + "...(redacted)"
		}
	}
	return strings.Join(fields, " ")
}

// ---------- server lifecycle ----------

func runServer(port int) {
	lg := log.New(os.Stderr, "claude2openai: ", log.LstdFlags)
	log.SetOutput(os.Stderr)
	log.SetPrefix("claude2openai: ")
	log.SetFlags(log.LstdFlags)

	if _, _, err := ensureFreshToken(); err != nil {
		lg.Printf("startup auth check failed: %v", err)
		lg.Printf("the proxy will keep running and retry on each request; run `codex login` if it persists")
	} else {
		lg.Printf("%s", authStatusLine())
	}

	srv := &http.Server{
		Addr:              fmt.Sprintf("127.0.0.1:%d", port),
		Handler:           newProxyServer().mux(),
		ReadHeaderTimeout: 30 * time.Second,
	}
	ln, err := net.Listen("tcp", srv.Addr)
	if err != nil {
		lg.Fatalf("cannot listen on %s: %v", srv.Addr, err)
	}

	if pf := portFilePath(); pf != "" {
		if err := os.WriteFile(pf, []byte(fmt.Sprintf("%d", port)), 0o600); err != nil {
			lg.Printf("warning: cannot write port file %s: %v", pf, err)
		}
	}
	lg.Printf("listening on http://%s (health: /health, messages: /v1/messages)", srv.Addr)
	lg.Printf("model tiers: main/opus/fable=%s sonnet=%s haiku=%s", mainModel(), codexModelSonnet, codexModelHaiku)
	if err := srv.Serve(ln); err != nil && err != http.ErrServerClosed {
		lg.Fatalf("server error: %v", err)
	}
}
