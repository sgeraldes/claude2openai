package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// claude2openai — run Anthropic-API clients (Claude Code) against OpenAI Codex
// models through the user's existing ChatGPT OAuth session from the Codex CLI.

const defaultPort = 3457

func usage() {
	fmt.Fprintf(os.Stderr, `claude2openai — Anthropic Messages proxy for Codex and Bedrock OpenAI

Usage:
  claude2openai server [port]             Run the Anthropic->Codex proxy (default port %d)
  claude2openai run [args...]             Start/attach to Codex and launch claude
  claude2openai test                      Test the Codex pipeline
  claude2openai --backend bedrock server [port]
  claude2openai --backend bedrock run [args...]
  claude2openai --backend bedrock test

Environment:
  CODEX_MODEL       Codex main model (default %s)
  BEDROCK_OPENAI=1  Select Bedrock Converse instead of Codex
  BEDROCK_MODEL     astra, sol, terra, luna, or a full inference profile id
  BEDROCK_SMALL_MODEL  Model for incoming haiku requests (default luna)
  BEDROCK_EFFORT       low, medium, high, or max; overrides Claude Code effort
                        and model defaults (astra/sol=high, terra=max, luna=medium)
`, defaultPort, codexDefaultModel)
}

func main() {
	args := os.Args[1:]
	backend := "codex"
	if bedrockEnabled() {
		backend = "bedrock"
	}
	if len(args) >= 2 && args[0] == "--backend" {
		backend = strings.ToLower(args[1])
		args = args[2:]
	}
	if len(args) < 1 {
		usage()
		os.Exit(2)
	}
	if backend == "bedrock" {
		runBedrockCommand(args)
		return
	}
	if backend != "codex" {
		fmt.Fprintf(os.Stderr, "unknown backend %q\n", backend)
		os.Exit(2)
	}
	switch args[0] {
	case "server":
		runServer(parsePort(args, defaultPort))
	case "run":
		runClaude(args[1:])
	case "test":
		runTest()
	case "help", "--help", "-h":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", args[0])
		usage()
		os.Exit(2)
	}
}

func runBedrockCommand(args []string) {
	switch args[0] {
	case "server":
		runBedrockServer(parsePort(args, defaultBedrockPort))
	case "run":
		runBedrockClaude(args[1:])
	case "test":
		runBedrockTest()
	case "help", "--help", "-h":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "bedrock backend supports server and test, got %q\n", args[0])
		os.Exit(2)
	}
}

func parsePort(args []string, fallback int) int {
	if len(args) < 2 {
		return fallback
	}
	port, err := strconv.Atoi(args[1])
	if err != nil || port <= 0 || port > 65535 {
		fmt.Fprintf(os.Stderr, "invalid port %q\n", args[1])
		os.Exit(2)
	}
	return port
}
