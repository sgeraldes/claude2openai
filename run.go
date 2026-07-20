package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// jsonString marshals s as a JSON string into a RawMessage (for string-form
// Anthropic content fields in tests and the built-in test request).
func jsonString(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

// run.go implements `claude2openai run`: attach to a running proxy or start one
// in-process, then launch claude with the Anthropic env pointed at it.

// probeHealth reports whether a healthy proxy answers on the given port.
func probeHealth(port int) bool {
	client := &http.Client{Timeout: 700 * time.Millisecond}
	resp, err := client.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// readPortFile returns the port recorded in ~/.claude2openai/proxy.port.
func readPortFile() int {
	pf := portFilePath()
	if pf == "" {
		return 0
	}
	data, err := os.ReadFile(pf)
	if err != nil {
		return 0
	}
	var p int
	if _, err := fmt.Sscanf(strings.TrimSpace(string(data)), "%d", &p); err != nil || p <= 0 || p > 65535 {
		return 0
	}
	return p
}

// findClaude resolves the claude binary, preferring ~/.local/bin/claude.exe
// over PATH and never settling for a .cmd shim when the native .exe exists
// alongside it (a stale shim once caused breakage on this machine).
func findClaude() (string, error) {
	home, _ := os.UserHomeDir()
	preferred := filepath.Join(home, ".local", "bin", "claude.exe")
	if fileExists(preferred) {
		return preferred, nil
	}
	path, err := exec.LookPath("claude")
	if err != nil {
		return "", fmt.Errorf("claude not found (looked at %s and PATH): %w", preferred, err)
	}
	lower := strings.ToLower(path)
	if strings.HasSuffix(lower, ".cmd") || strings.HasSuffix(lower, ".bat") {
		exe := strings.TrimSuffix(path, filepath.Ext(path)) + ".exe"
		if fileExists(exe) {
			return exe, nil
		}
	}
	return path, nil
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// runClaude implements the run command.
func runClaude(args []string) {
	port := defaultPort
	if pf := readPortFile(); pf > 0 {
		port = pf
	}

	startedProxy := false
	if probeHealth(port) {
		fmt.Fprintf(os.Stderr, "claude2openai: attaching to proxy on 127.0.0.1:%d\n", port)
	} else {
		// Start the proxy in-process; it stops when this command exits.
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
		if err != nil {
			fmt.Fprintf(os.Stderr, "claude2openai: cannot start proxy on port %d: %v\n", port, err)
			os.Exit(1)
		}
		srv := &http.Server{Handler: newProxyServer().mux(), ReadHeaderTimeout: 30 * time.Second}
		go func() { _ = srv.Serve(ln) }()
		startedProxy = true

		// Wait until it answers health checks (should be immediate).
		deadline := time.Now().Add(5 * time.Second)
		for !probeHealth(port) && time.Now().Before(deadline) {
			time.Sleep(50 * time.Millisecond)
		}
		if !probeHealth(port) {
			fmt.Fprintf(os.Stderr, "claude2openai: proxy did not become healthy on port %d\n", port)
			os.Exit(1)
		}
		if pf := portFilePath(); pf != "" {
			_ = os.WriteFile(pf, []byte(fmt.Sprintf("%d", port)), 0o600)
		}
		fmt.Fprintf(os.Stderr, "claude2openai: proxy started on 127.0.0.1:%d\n", port)
	}

	claudePath, err := findClaude()
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai: %v\n", err)
		os.Exit(1)
	}

	cmd := exec.Command(claudePath, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("ANTHROPIC_BASE_URL=http://127.0.0.1:%d", port),
		"ANTHROPIC_AUTH_TOKEN=claude2openai",
	)

	runErr := cmd.Run()

	if startedProxy {
		// The in-process proxy dies with this process; nothing else to do.
		fmt.Fprintf(os.Stderr, "claude2openai: claude exited, stopping proxy\n")
	}
	if runErr == nil {
		return
	}
	if exitErr, ok := runErr.(*exec.ExitError); ok {
		os.Exit(exitErr.ExitCode())
	}
	fmt.Fprintf(os.Stderr, "claude2openai: failed to launch claude: %v\n", runErr)
	os.Exit(1)
}

// runTest sends a tiny request through the full pipeline (auth -> translate ->
// upstream stream -> aggregate) and prints the model's reply. Secret-safe.
func runTest() {
	fmt.Println(authStatusLine())

	token, accountID, err := ensureFreshToken()
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai test: %v\n", err)
		os.Exit(1)
	}
	_ = token // only used inside the upstream call below

	reqBody := &anthropicRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []anthropicMessage{
			{Role: "user", Content: jsonString("Reply with exactly: CODEX OK")},
		},
		System: jsonString("You are a concise assistant."),
	}
	upstream := translateRequest(reqBody)
	payload, err := json.Marshal(upstream)
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai test: encode: %v\n", err)
		os.Exit(1)
	}

	httpReq, err := http.NewRequest(http.MethodPost, codexResponsesURL, bytes.NewReader(payload))
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai test: %v\n", err)
		os.Exit(1)
	}
	httpReq.Header.Set("Authorization", "Bearer "+token)
	httpReq.Header.Set("chatgpt-account-id", accountID)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("originator", codexOriginator)
	httpReq.Header.Set("OpenAI-Beta", "responses=experimental")
	httpReq.Header.Set("User-Agent", codexUserAgent)

	fmt.Printf("upstream model: %s\n", upstream.Model)
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai test: request failed: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		fmt.Fprintf(os.Stderr, "claude2openai test: backend returned %s: %s\n", resp.Status, truncate(string(body), 500))
		os.Exit(1)
	}

	translator := newStreamTranslator(reqBody.Model, false)
	var events []anthropicEvent
	for ev := range parseResponsesSSE(resp.Body) {
		if ev.parseErr != nil {
			continue
		}
		events = append(events, translator.Handle(ev.event)...)
	}
	events = append(events, translator.Finalize()...)
	msg, err := aggregateEvents(events, reqBody.Model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "claude2openai test: %v\n", err)
		os.Exit(1)
	}
	for _, block := range msg.Content {
		if block["type"] == "text" {
			fmt.Printf("reply: %s\n", block["text"])
		}
	}
	fmt.Printf("usage: %d input / %d output tokens\n", msg.Usage["input_tokens"], msg.Usage["output_tokens"])
	fmt.Println("test: OK")
}
