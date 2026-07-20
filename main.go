package main

import (
	"fmt"
	"os"
	"strconv"
)

// claude2openai — run Anthropic-API clients (Claude Code) against OpenAI Codex
// models through the user's existing ChatGPT OAuth session from the Codex CLI.

const defaultPort = 3457

func usage() {
	fmt.Fprintf(os.Stderr, `claude2openai — Claude Code on OpenAI Codex (ChatGPT OAuth)

Usage:
  claude2openai server [port]   Run the Anthropic->Codex proxy (default port %d)
  claude2openai run [args...]   Start/attach to the proxy and launch claude with args
  claude2openai test            Send a tiny request through the pipeline and print the reply

Environment:
  CODEX_MODEL   Override the default Codex model (default %s)

Auth is read from %%USERPROFILE%%\.codex\auth.json (ChatGPT OAuth, read-only
except careful token-refresh write-back). Never prints token values.
`, defaultPort, codexDefaultModel)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "server":
		port := defaultPort
		if len(os.Args) >= 3 {
			p, err := strconv.Atoi(os.Args[2])
			if err != nil || p <= 0 || p > 65535 {
				fmt.Fprintf(os.Stderr, "invalid port %q\n", os.Args[2])
				os.Exit(2)
			}
			port = p
		}
		runServer(port)
	case "run":
		runClaude(os.Args[2:])
	case "test":
		runTest()
	case "help", "--help", "-h":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}
