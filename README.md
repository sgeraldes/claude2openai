<p align="center">
  <img src="docs/images/claude2openai-hero.png" alt="claude2openai" width="720" />
</p>

<h1 align="center">claude2openai</h1>

<p align="center">
  Claude Code on OpenAI's Codex models — through your existing ChatGPT subscription. No API key needed.
</p>

<p align="center">
  <a href="#commands">Commands</a> •
  <a href="#install">Install</a> •
  <a href="#model-tiers">Model tiers</a> •
  <a href="#how-auth-works">Auth</a> •
  <a href="#troubleshooting">Troubleshooting</a>
</p>

> **Part of [claude2all](https://github.com/sgeraldes/claude2all)** — one isolated Claude Code launcher per backend: Kimi K3, AWS Bedrock, OpenAI (this project), Kiro, and multiple claude.ai accounts.

Run Anthropic-API clients — primarily Claude Code — against OpenAI Codex models
using your existing **ChatGPT OAuth session from the Codex CLI**. No OpenAI API
key needed.

`claude2openai` is a localhost proxy that speaks the Anthropic Messages API on
the client side (`POST /v1/messages`, SSE streaming, tool use) and the OpenAI
Responses API on the upstream side (`POST https://chatgpt.com/backend-api/codex/responses`),
authenticating with the OAuth tokens the Codex CLI already stores at
`~/.codex/auth.json`.

> **Naming note:** this tool is `claude2openai`, but the Codex CLI's own files
> keep their names — OAuth tokens are read from **`~/.codex/auth.json`** and the
> recovery command really is **`codex login`**. Only claude2openai's own files
> (binary, wrappers, `~/.claude2openai/`, `~/.claude-profiles/openai/`) use the
> new name.

## Commands

```
claude2openai server [port]   Run the proxy (default port 3457).
                              Writes the port to ~/.claude2openai/proxy.port.
                              GET /health returns 200 OK.
claude2openai run [args...]   Start the proxy (or attach to a running one via
                              proxy.port + /health), then launch `claude` with
                              ANTHROPIC_BASE_URL=http://127.0.0.1:<port> and
                              ANTHROPIC_AUTH_TOKEN=claude2openai. All args are
                              forwarded. If this command started the proxy, the
                              proxy stops when claude exits.
claude2openai test            Send a tiny request through the whole pipeline
                              (auth -> translate -> upstream -> aggregate) and
                              print the model's reply. Safe: never prints tokens.
```

## Install

The binary the wrappers use lives at `~/.claude2openai/claude2openai.exe`:

```bash
cd /g/code/claude2openai
go build -ldflags "-s -w" -o claude2openai.exe .
cp claude2openai.exe ~/.claude2openai/claude2openai.exe
```

Two launchers live in `~/.local/bin/` (both on PATH):

- `claude2openai` — Git Bash wrapper: isolated profile
  (`~/.claude-profiles/openai`), seeds onboarding/trust/permissions like the
  other claude2* wrappers, pins every Claude model slot to a gpt-5.6 variant,
  then `run`. `server`/`test` subcommands pass straight through.
- `claude2openai.cmd` — the same for cmd/PowerShell (delegates to the bash
  wrapper; claude.exe must be launched through real Git Bash on this machine).

```bash
claude2openai                      # interactive Claude Code on Codex
claude2openai -p "fix the typo in README.md"
claude2openai server 3457          # persistent proxy; stop with Ctrl+C
```

## Model tiers

Claude model ids are mapped to gpt-5.6 variants by tier:

| Incoming model                          | Codex model      |
|-----------------------------------------|------------------|
| main / opus / fable (and any fallback)  | `gpt-5.6-sol`    |
| sonnet (and subagents)                  | `gpt-5.6-terra`  |
| haiku                                   | `gpt-5.6-luna`   |

(`terra` is the backend's exact spelling — verified against the live catalog,
it is *not* "tierra".)

Ids that already look like gpt/codex models (`gpt-*`, anything containing
`codex`) pass through unchanged. The wrapper pins `ANTHROPIC_MODEL`,
`ANTHROPIC_DEFAULT_{FABLE,OPUS,SONNET,HAIKU}_MODEL` and
`CLAUDE_CODE_SUBAGENT_MODEL` to these variants so Claude Code's built-in
claude-* ids never leak through. Overrides: `CODEX_MODEL` (main tier, also
read by the proxy), `CODEX_SONNET_MODEL`, `CODEX_HAIKU_MODEL`,
`CODEX_SUBAGENT_MODEL` (wrapper only).

Models served by the account (verified via `GET /backend-api/codex/models`):
`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`,
`gpt-5.4-mini`, `gpt-5.3-codex-spark`, `codex-auto-review`.

## How auth works

- Reads `~/.codex/auth.json` (the **Codex CLI's** ChatGPT OAuth file:
  `access_token`, `refresh_token`, `account_id`). The file is treated as
  read-only **except** careful token-refresh write-back.
- When the access token is expired (or rejected with a 401), the proxy
  refreshes it against `https://auth.openai.com/oauth/token` (the Codex CLI's
  public OAuth client id), then writes the new tokens back to `auth.json`
  **preserving every other field**, after saving a timestamped backup to
  `~/.claude2openai/`.
- Token values are never printed or logged — only lengths and expiry times.
- If refresh fails, requests fail with a clear error telling you to run
  `codex login` again.

## Translation notes / limits

- Messages: user/assistant text, base64 images, `tool_use` and `tool_result`
  blocks all map to their Responses API equivalents (`function_call`,
  `function_call_output`).
- The upstream call is always `stream: true, store: false` (the backend
  requires streaming); non-streaming client requests are aggregated from the
  stream.
- `max_tokens` is **dropped**: the backend rejects `max_output_tokens` (400).
- `temperature`, `top_p`, `top_k`, `stop_sequences` are not supported by the
  Codex backend and are dropped.
- The reasoning effort sent to Codex is the first valid value of: `CODEX_EFFORT`
  on the proxy (applies to every request), the request's `effort`, its
  `output_config.effort` (what `CLAUDE_CODE_EFFORT_LEVEL` and `/effort` set in
  Claude Code), and, only when the request asks for thinking,
  `CLAUDE_CODE_EFFORT_LEVEL` in the proxy's own environment and the `thinking`
  token budget (under 2,048 low, 15,000 or more high, medium otherwise). An
  invalid value is logged and the next one is tried. The backend accepts `none`,
  `minimal`, `low`, `medium`, `high`, `xhigh` and `max` (its 400 for anything else
  says so), so every Claude Code level passes as is; the `messages:` log line
  shows the effort sent. Before 25-Sep-2026 only the budget was read, and Claude
  Code sends `thinking: {"type":"adaptive"}` with no budget, so every run went out
  at medium whatever level it asked for.
- Reasoning summaries (summary `auto`) stream back as Anthropic thinking blocks
  when the request asked for thinking; otherwise they are dropped, even when
  `CODEX_EFFORT` made the model reason.
- `/v1/messages/count_tokens` returns a local ~4-chars-per-token estimate.

## Backend Bedrock (modelos OpenAI)

El mismo binario también puede exponer Anthropic Messages sobre Amazon Bedrock
ConverseStream para los modelos OpenAI. Este camino es necesario porque el modo
Bedrock nativo de Claude Code usa el contrato Anthropic de InvokeModel; Astra,
Sol, Terra y Luna usan Converse.

```bash
export AWS_PROFILE=dfx5-dfx5-internal-apps-dev-administratoraccess
export AWS_REGION=us-west-2
BEDROCK_MODEL=astra claude2openai --backend bedrock test
BEDROCK_MODEL=sol   claude2openai --backend bedrock test
BEDROCK_MODEL=terra BEDROCK_EFFORT=max claude2openai --backend bedrock test
claude2openai --backend bedrock server 3458
```

El launcher recomendado es `claude2bedrock --openai`, incluido en `claude2all`.
Inicia el proxy dentro del proceso, configura `ANTHROPIC_BASE_URL`, usa el perfil
aislado `~/.claude-profiles/bedrock-openai` y cierra el listener al terminar.

| Valor de `BEDROCK_MODEL` | Inference profile |
|---|---|
| `astra` (default) | `us.openai.gpt-6-astra` |
| `sol` | `us.openai.gpt-5.6-sol` |
| `terra` | `us.openai.gpt-5.6-terra` |
| `luna` | `us.openai.gpt-5.6-luna` |

También acepta un inference profile ID completo. `BEDROCK_SMALL_MODEL` selecciona
el modelo para solicitudes cuyo ID entrante contiene `haiku` (default `luna`).
`AWS_PROFILE` y `AWS_REGION` controlan las credenciales SigV4; perfiles SSO son
compatibles mediante el credential chain de AWS SDK for Go v2.

### Esfuerzo de razonamiento

Para estos perfiles OpenAI, ConverseStream acepta el campo adicional
`{"reasoning":{"effort":"<valor>"}}`. La forma plana
`{"reasoning_effort":"<valor>"}` fue rechazada por Astra, Sol, Terra y Luna con
`unknown_parameter` el 2026-09-14. Los cuatro perfiles aceptaron `low`, `medium`,
`high` y `max` mediante llamadas reales.

La precedencia es `BEDROCK_EFFORT` > `effort` / `output_config.effort` de la
solicitud Anthropic > `CLAUDE_CODE_EFFORT_LEVEL` > `thinking` de la solicitud >
default del modelo. `xhigh` de Claude Code se normaliza a `high` porque Bedrock
sólo acepta los cuatro valores anteriores. Defaults: Astra `high`, Sol `high`,
Terra `max`, Luna `medium`.

```bash
claude2bedrock --openai --model luna --effort medium -p "revisa el deploy"
claude2bedrock --openai --model terra --effort max -p "implementa el cambio"
claude2bedrock --openai test --model terra --effort max
```

El log de cada request incluye el modelo resuelto y el esfuerzo enviado. El
subcomando `test` imprime ambos antes de la respuesta.

### Validación real (2026-09-14)

| Perfil | Default enviado | Respuesta de `test` |
|---|---|---|
| `us.openai.gpt-6-astra` | `high` | `BEDROCK OPENAI OK` |
| `us.openai.gpt-5.6-sol` | `high` | `BEDROCK OPENAI OK` |
| `us.openai.gpt-5.6-terra` | `max` | `BEDROCK OPENAI OK` |
| `us.openai.gpt-5.6-luna` | `medium` | `BEDROCK OPENAI OK` |

Cada perfil también aceptó una llamada separada con cada valor: `low`, `medium`,
`high` y `max`.

La traducción incluye system, texto, imágenes base64, tools, `tool_use`,
`tool_result` (incluido estado de error), tool choice, `max_tokens` y
`stop_sequences`. `temperature` y `top_p` se omiten porque GPT-5.6 los rechaza.
Las respuestas streaming y no streaming se convierten al contrato Anthropic,
incluido uso y stop reason. Limitaciones: `count_tokens` es una estimación local
de caracteres/4; los bloques thinking se descartan (se registra una advertencia
una vez); prompt caching no está implementado.

## Troubleshooting

- **`codex token refresh failed ... run `codex login` again`** — your ChatGPT
  session expired; run `codex login`, then retry.
- **401 from the backend even after refresh** — run `codex login` again; make
  sure `codex` itself works (`codex exec "say hi"`).
- **Port already in use** — another proxy is running; `run` will attach to it
  automatically, or pick another port: `claude2openai server 3458`.
- **Check the pipeline** — `claude2openai test` prints the model's reply to a
  canned prompt plus (masked) auth status.

## Development

```bash
go vet ./...
go test ./...
go build -ldflags "-s -w" -o claude2openai.exe .
```

Dependencies: AWS SDK for Go v2 for Bedrock ConverseStream, SigV4 and SSO credentials.
