package main

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// auth.go reads (and, only when expired, refreshes) the Codex CLI's ChatGPT
// OAuth session at ~/.codex/auth.json. Token values are never logged.

const (
	codexOAuthTokenURL = "https://auth.openai.com/oauth/token"
	codexOAuthClientID = "app_EMoamEEZ73f0CkXaXp7hrann" // codex_cli_rs public client id
	// Refresh when the access token expires within this window.
	refreshWindow = 2 * time.Minute
)

// codexTokens mirrors the tokens object of ~/.codex/auth.json.
type codexTokens struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	AccountID    string `json:"account_id"`
}

type codexAuthFile struct {
	AuthMode    string      `json:"auth_mode"`
	Tokens      codexTokens `json:"tokens"`
	LastRefresh string      `json:"last_refresh"`
}

var (
	authMu     sync.Mutex // serializes refresh + write-back
	cachedAuth *codexAuthFile
)

func codexAuthPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".codex", "auth.json")
}

func stateDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	dir := filepath.Join(home, ".claude2openai")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	return dir, nil
}

func portFilePath() string {
	dir, err := stateDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "proxy.port")
}

// loadAuth reads ~/.codex/auth.json. It never prints token material.
func loadAuth() (*codexAuthFile, error) {
	path := codexAuthPath()
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read %s: %w (run `codex login` first)", path, err)
	}
	var a codexAuthFile
	if err := json.Unmarshal(data, &a); err != nil {
		return nil, fmt.Errorf("cannot parse %s: %w", path, err)
	}
	if a.AuthMode != "chatgpt" {
		return nil, fmt.Errorf("%s has auth_mode %q, expected \"chatgpt\" (run `codex login` with ChatGPT)", path, a.AuthMode)
	}
	if a.Tokens.AccessToken == "" {
		return nil, fmt.Errorf("%s has no access token (run `codex login` again)", path)
	}
	return &a, nil
}

// jwtExpiry extracts the exp claim from a JWT without verifying the signature
// (the backend verifies it; we only need the expiry for refresh scheduling).
func jwtExpiry(token string) (time.Time, bool) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return time.Time{}, false
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return time.Time{}, false
	}
	var claims struct {
		Exp int64 `json:"exp"`
	}
	if err := json.Unmarshal(payload, &claims); err != nil || claims.Exp == 0 {
		return time.Time{}, false
	}
	return time.Unix(claims.Exp, 0), true
}

// ensureFreshToken returns a valid access token + account id, refreshing via
// the OAuth refresh token when the access token is expired or nearly so.
// The error message deliberately tells the user how to recover.
func ensureFreshToken() (accessToken, accountID string, err error) {
	authMu.Lock()
	defer authMu.Unlock()

	a, err := loadAuth()
	if err != nil {
		return "", "", err
	}

	exp, ok := jwtExpiry(a.Tokens.AccessToken)
	if ok && time.Until(exp) > refreshWindow {
		cachedAuth = a
		return a.Tokens.AccessToken, a.Tokens.AccountID, nil
	}

	// Token expired (or expiry unknown): refresh it.
	if a.Tokens.RefreshToken == "" {
		return "", "", fmt.Errorf("codex access token is expired and no refresh token is available; run `codex login` again")
	}
	if err := refreshAndStore(a); err != nil {
		return "", "", fmt.Errorf("codex token refresh failed (%v); run `codex login` again to renew your ChatGPT session", err)
	}
	cachedAuth = a
	return a.Tokens.AccessToken, a.Tokens.AccountID, nil
}

// forceRefresh refreshes the OAuth tokens even if the access token still looks
// valid. Used when the backend rejects a seemingly valid token with a 401.
func forceRefresh() error {
	authMu.Lock()
	defer authMu.Unlock()
	a, err := loadAuth()
	if err != nil {
		return err
	}
	if a.Tokens.RefreshToken == "" {
		return fmt.Errorf("no refresh token available")
	}
	if err := refreshAndStore(a); err != nil {
		return err
	}
	cachedAuth = a
	return nil
}

// refreshAndStore exchanges the refresh token for new tokens and writes them
// back to auth.json, preserving all other fields. Caller must hold authMu.
func refreshAndStore(a *codexAuthFile) error {
	refreshed, err := refreshTokens(a.Tokens.RefreshToken)
	if err != nil {
		return err
	}
	a.Tokens.AccessToken = refreshed.AccessToken
	if refreshed.RefreshToken != "" {
		a.Tokens.RefreshToken = refreshed.RefreshToken
	}
	if refreshed.IDToken != "" {
		a.Tokens.IDToken = refreshed.IDToken
	}
	if err := writeBackAuth(a); err != nil {
		// Non-fatal for this process: we have a working token in memory, but
		// warn so a persistent write failure is visible.
		fmt.Fprintf(os.Stderr, "claude2openai: warning: refreshed token could not be written back to auth.json: %v\n", err)
	}
	return nil
}

type refreshResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token"`
	ExpiresIn    int    `json:"expires_in"`
	TokenType    string `json:"token_type"`
}

func refreshTokens(refreshToken string) (*refreshResponse, error) {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"client_id":     {codexOAuthClientID},
		"refresh_token": {refreshToken},
	}
	req, err := http.NewRequest(http.MethodPost, codexOAuthTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if resp.StatusCode != http.StatusOK {
		// Do not include the response body verbatim if it might echo the
		// refresh token; oauth errors are short JSON without secrets.
		return nil, fmt.Errorf("oauth token endpoint returned %s: %s", resp.Status, truncate(string(body), 300))
	}
	var rr refreshResponse
	if err := json.Unmarshal(body, &rr); err != nil {
		return nil, fmt.Errorf("cannot parse refresh response: %w", err)
	}
	if rr.AccessToken == "" {
		return nil, fmt.Errorf("refresh response contained no access token")
	}
	return &rr, nil
}

// writeBackAuth updates auth.json with refreshed tokens while preserving every
// other field of the file. It writes a timestamped backup under
// ~/.claude2openai/ first, then replaces the file atomically.
func writeBackAuth(updated *codexAuthFile) error {
	path := codexAuthPath()
	raw, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	// Merge into the original JSON tree so unknown fields survive.
	var tree map[string]any
	if err := json.Unmarshal(raw, &tree); err != nil {
		return fmt.Errorf("cannot re-parse auth.json for write-back: %w", err)
	}
	tokens, _ := tree["tokens"].(map[string]any)
	if tokens == nil {
		tokens = map[string]any{}
	}
	tokens["access_token"] = updated.Tokens.AccessToken
	tokens["refresh_token"] = updated.Tokens.RefreshToken
	tokens["id_token"] = updated.Tokens.IDToken
	tokens["account_id"] = updated.Tokens.AccountID
	tree["tokens"] = tokens
	tree["last_refresh"] = time.Now().UTC().Format(time.RFC3339Nano)

	// Backup (outside ~/.codex so we only ever modify auth.json itself there).
	if dir, derr := stateDir(); derr == nil {
		backup := filepath.Join(dir, fmt.Sprintf("auth.json.backup-%d", time.Now().Unix()))
		if berr := os.WriteFile(backup, raw, 0o600); berr != nil {
			return fmt.Errorf("backup before write-back failed: %w", berr)
		}
	}

	out, err := json.MarshalIndent(tree, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".claude2openai-tmp"
	if err := os.WriteFile(tmp, out, 0o600); err != nil {
		return err
	}
	// Windows cannot rename over an existing file; remove then rename.
	// The backup above makes this recoverable.
	if err := os.Remove(path); err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, path); err != nil {
		// Restore from backup content we still hold in memory.
		_ = os.WriteFile(path, raw, 0o600)
		os.Remove(tmp)
		return err
	}
	return nil
}

// maskToken renders a safe, non-reversible description of a token for logs.
func maskToken(t string) string {
	if t == "" {
		return "(none)"
	}
	return fmt.Sprintf("(hidden, len %d)", len(t))
}

// authStatusLine returns a human-readable, secret-free auth summary.
func authStatusLine() string {
	a, err := loadAuth()
	if err != nil {
		return fmt.Sprintf("auth: %v", err)
	}
	exp, ok := jwtExpiry(a.Tokens.AccessToken)
	expStr := "unknown expiry"
	if ok {
		expStr = "expires " + exp.UTC().Format(time.RFC3339)
	}
	return fmt.Sprintf("auth: chatgpt mode, access token %s %s, account id (len %d)",
		maskToken(a.Tokens.AccessToken), expStr, len(a.Tokens.AccountID))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
