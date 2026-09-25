package main

import (
	"os"
	"testing"
)

// The proxy reads its model and effort from the environment, and whatever launches the tests (a
// shell that ran claude2openai, a delegated review) may export them. Every test starts from none of
// them set and sets only what it checks.
func TestMain(m *testing.M) {
	for _, name := range []string{"CODEX_MODEL", "CODEX_EFFORT", "CLAUDE_CODE_EFFORT_LEVEL",
		"BEDROCK_OPENAI", "BEDROCK_MODEL", "BEDROCK_SMALL_MODEL", "BEDROCK_EFFORT"} {
		os.Unsetenv(name)
	}
	os.Exit(m.Run())
}
