package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func makeJWT(exp int64) string {
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256"}`))
	payload := base64.RawURLEncoding.EncodeToString([]byte(fmt.Sprintf(`{"exp":%d,"sub":"test"}`, exp)))
	return header + "." + payload + ".fakesig"
}

func TestJwtExpiry(t *testing.T) {
	exp := time.Now().Add(time.Hour).Unix()
	got, ok := jwtExpiry(makeJWT(exp))
	if !ok {
		t.Fatal("expected valid JWT")
	}
	if got.Unix() != exp {
		t.Errorf("exp = %d, want %d", got.Unix(), exp)
	}
	if _, ok := jwtExpiry("not-a-jwt"); ok {
		t.Error("garbage should not parse")
	}
	if _, ok := jwtExpiry("opaque-token-no-dots"); ok {
		t.Error("opaque token should not parse as JWT")
	}
}

// setupFakeHome points os.UserHomeDir at a temp dir with a .codex/auth.json.
func setupFakeHome(t *testing.T, authJSON string) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("USERPROFILE", home)
	t.Setenv("HOME", home)
	codexDir := filepath.Join(home, ".codex")
	if err := os.MkdirAll(codexDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(codexDir, "auth.json"), []byte(authJSON), 0o600); err != nil {
		t.Fatal(err)
	}
	return home
}

func TestLoadAuthValidatesShape(t *testing.T) {
	setupFakeHome(t, `{"auth_mode":"apikey","tokens":{}}`)
	if _, err := loadAuth(); err == nil || !strings.Contains(err.Error(), "auth_mode") {
		t.Errorf("non-chatgpt auth should be rejected: %v", err)
	}

	setupFakeHome(t, `{"auth_mode":"chatgpt","tokens":{"id_token":"i","access_token":"","refresh_token":"r","account_id":"a"}}`)
	if _, err := loadAuth(); err == nil || !strings.Contains(err.Error(), "access token") {
		t.Errorf("missing access token should be rejected: %v", err)
	}
}

func TestEnsureFreshTokenErrorMentionsCodexLogin(t *testing.T) {
	// Expired access token + no refresh token => clear recovery instruction.
	setupFakeHome(t, fmt.Sprintf(`{
		"auth_mode":"chatgpt",
		"tokens":{"id_token":"i","access_token":%q,"refresh_token":"","account_id":"a"},
		"last_refresh":"2020-01-01T00:00:00Z"
	}`, makeJWT(time.Now().Add(-time.Hour).Unix())))

	_, _, err := ensureFreshToken()
	if err == nil {
		t.Fatal("expected an error for an expired token without refresh token")
	}
	if !strings.Contains(err.Error(), "codex login") {
		t.Errorf("error must tell the user to run `codex login`: %v", err)
	}
}

func TestEnsureFreshTokenUsesValidToken(t *testing.T) {
	valid := makeJWT(time.Now().Add(time.Hour).Unix())
	setupFakeHome(t, fmt.Sprintf(`{
		"auth_mode":"chatgpt",
		"tokens":{"id_token":"i","access_token":%q,"refresh_token":"r","account_id":"acct-123"},
		"last_refresh":"2020-01-01T00:00:00Z"
	}`, valid))

	tok, acct, err := ensureFreshToken()
	if err != nil {
		t.Fatalf("valid token should pass: %v", err)
	}
	if tok != valid || acct != "acct-123" {
		t.Errorf("got (len %d, %q)", len(tok), acct)
	}
}

func TestWriteBackAuthPreservesFields(t *testing.T) {
	home := setupFakeHome(t, `{
		"auth_mode":"chatgpt",
		"OPENAI_API_KEY":null,
		"tokens":{"id_token":"old-id","access_token":"old-access","refresh_token":"old-refresh","account_id":"acct-1"},
		"last_refresh":"2020-01-01T00:00:00Z",
		"some_future_field":{"nested":[1,2,3]}
	}`)

	updated := &codexAuthFile{
		Tokens: codexTokens{
			IDToken:      "new-id",
			AccessToken:  "new-access",
			RefreshToken: "new-refresh",
			AccountID:    "acct-1",
		},
	}
	if err := writeBackAuth(updated); err != nil {
		t.Fatalf("writeBackAuth: %v", err)
	}

	raw, err := os.ReadFile(filepath.Join(home, ".codex", "auth.json"))
	if err != nil {
		t.Fatal(err)
	}
	var tree map[string]any
	if err := json.Unmarshal(raw, &tree); err != nil {
		t.Fatal(err)
	}
	tokens := tree["tokens"].(map[string]any)
	if tokens["access_token"] != "new-access" || tokens["refresh_token"] != "new-refresh" || tokens["id_token"] != "new-id" {
		t.Errorf("tokens not updated: %v", tokens)
	}
	if tree["auth_mode"] != "chatgpt" {
		t.Errorf("auth_mode lost: %v", tree["auth_mode"])
	}
	if _, present := tree["OPENAI_API_KEY"]; !present {
		t.Error("OPENAI_API_KEY field lost")
	}
	future, ok := tree["some_future_field"].(map[string]any)
	if !ok {
		t.Fatal("unknown future field lost")
	}
	if len(future["nested"].([]any)) != 3 {
		t.Errorf("nested future data mangled: %v", future)
	}
	lr, _ := tree["last_refresh"].(string)
	if lr == "2020-01-01T00:00:00Z" || lr == "" {
		t.Errorf("last_refresh should be updated: %q", lr)
	}

	// A backup must exist under the fake ~/.claude2openai.
	backups, _ := filepath.Glob(filepath.Join(home, ".claude2openai", "auth.json.backup-*"))
	if len(backups) == 0 {
		t.Error("no backup written before write-back")
	}
}

func TestSanitizeUpstreamErrorRedactsLongRuns(t *testing.T) {
	long := strings.Repeat("A", 200)
	err := sanitizeUpstreamError(fmt.Errorf("backend returned 400: token %s was invalid", long))
	if strings.Contains(err, long) {
		t.Errorf("long secret-looking run must be redacted: %s", err)
	}
	if !strings.Contains(err, "(redacted)") {
		t.Errorf("expected redaction marker: %s", err)
	}
}

func TestMaskToken(t *testing.T) {
	secret := "super-secret-token-value"
	if got := maskToken(secret); strings.Contains(got, secret) {
		t.Errorf("maskToken leaked the token: %q", got)
	}
	if got := maskToken(""); got != "(none)" {
		t.Errorf("empty token: %q", got)
	}
}

func TestSanitizeToolName(t *testing.T) {
	cases := map[string]string{
		"Bash":          "Bash",
		"mcp__x__y":     "mcp__x__y",
		"bad name!":     "bad_name_",
		"tool.with.dots": "tool_with_dots",
		"":              "tool",
	}
	for in, want := range cases {
		if got := sanitizeToolName(in); got != want {
			t.Errorf("sanitizeToolName(%q) = %q, want %q", in, got, want)
		}
	}
	long := strings.Repeat("x", 100)
	if got := sanitizeToolName(long); len(got) != 64 {
		t.Errorf("names must cap at 64 chars, got %d", len(got))
	}
}
