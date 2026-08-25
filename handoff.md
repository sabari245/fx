# Handoff: OpenAI-compatible endpoint provider for fx

This document is a complete record of the work performed in this session, from initial exploration through implementation, testing, and bug-fix. It is written for the next developer (or agent) to pick up exactly where this left off.

## Branch

- Branch: **`feat/openai-compatibility`**
- Base: `main` at `16eda25` (fast-forwarded from `fff3f63` at session start)
- Status: in-progress skeleton, **not** declared merge-ready (Full CI has not run)

## Goal

Make fx work against **any OpenAI-compatible endpoint** (vLLM, Ollama, LM Studio, llama.cpp, LiteLLM, DeepSeek, OpenRouter, Groq, etc.) that speaks the **Chat Completions** wire format (`POST /v1/chat/completions`, SSE `choices[0].delta`, terminated by `data: [DONE]`).

Constraints established with the user:

- New features must be **additive** (new files + isolated small edits) so upstream `main` merges stay clean.
- No third-party deps outside the Zig stdlib without discussion.
- Do not hand-roll every feature from scratch. Research a mature library first.

## Background research (done before writing code)

### fx provider architecture

fx composes a `provider_set.Set` of `Bundle`s in `builtins/providers.zig`. Each `Bundle` (`src/core/gateway/provider_set.zig`) wires optional capability slots:

- `.agent_stream` — the core extension point (`src/core/agent/stream_provider.zig`): a `stream_fn` taking a typed `ModelRequest` and returning a `Result` of streaming `Event`s (`content_delta`, `reasoning_delta`, `tool_started`, `tool_input_delta`).
- `.cli_model_catalog` / `.model_catalog` — model list + capability metadata (`src/core/gateway/model_catalog.zig`).
- `.permission_reviewer` — auto-review for `auto` mode (`src/core/permissions/auto_classifier.zig`). **Nullable**; a null reviewer disables auto-review and fails closed.
- `.deferred_usage` / `.credits` / `.fx_search`.

Provider identity is a `ProviderId` enum (`src/core/config/model_provider.zig`), aliased in `src/core/auth/provider_catalog.zig`, keyed into `model_preferences`, and resolved through `credentials.resolveForProvider`.

The existing providers speak the **OpenAI Responses API** (`responses_protocol.zig`, `/v3/ai/language-model`, `/v1/responses`), not Chat Completions. So this is a new transport, not a URL override.

### Library research verdict

Searched the Zig ecosystem (GitHub API + web):

- `0xble/zig-openai`, `chad/zig-openai` — **no longer exist** (404).
- `zouyee/llmlite` — 11★, stale (~4 months), `NOASSERTION` license.
- `Kludex/zigai` — 3★, created 2 weeks ago.
- `lightpanda-io/zenai` — 9★, **actively maintained** (pushed same day), Apache-2.0, **requires Zig ≥ 0.16.0**, covers SSE streaming, function calling, model listing, and OpenAI-compatible backends via `OPENAI_BASE_URL`.

**Verdict:** `zenai` is the only viable candidate but plugging it in is *not* free — it needs a full translation layer from fx's `stream_provider` contract (`ModelRequest`/`EventSink`/`ModelCompletion`/`ToolSelection`/`DeliveryCertainty`/`Admission`/cancellation/bounded-I/O/`auto_classifier`) onto zenai's types, which is roughly the same code volume as writing a thin transport, plus a new dependency and pinning risk. The user chose the **thin in-repo adapter** (stdlib-only, keeps upstream merges clean). `zenai`'s README was used as the wire-format reference.

## Implementation

### New file: `src/gateway/openai_compat.zig` (1058 lines)

A self-contained Chat Completions transport modeled on `src/gateway/xai_grok.zig` (the standalone Responses-API provider). It reuses fx's existing `gateway/client.zig` for bounded HTTP I/O, cancellation watchers, and network-failure evidence, and reuses `model_tool_schema` for tool-schema serialization.

Provides:

- `agent_stream_provider` — the `stream_fn` (builds payload, POSTs, parses SSE).
- `buildRequest` — serializes `RequestData` into a Chat Completions body:
  - `model`, `stream: true`, `tool_choice` (`auto`/`none`/`required`), `messages`, `max_tokens`, `reasoning_effort`, `response_format` (json_schema).
  - `messages`: system/user/assistant/tool roles; assistant tool_calls as `{"tool_calls":[{id,type:"function",function:{name,arguments}}]}`; tool results as `{"role":"tool","tool_call_id","content"}`.
  - vision: `verified_images` base64-encoded into `{"type":"image_url","image_url":{"url":"data:<media>;base64,..."}}` content parts on the last user message.
  - tools: `advertised_names`/`additional_functions`/`selected_dynamic` → `{"type":"function","function":{name,description,parameters}}`.
- `streamPrepared` — HTTP POST to `{base}/chat/completions`, bearer auth when a key is set, bounded connect, SSE read.
- SSE reducer (`consumeSse`/`applyJson`/`finalize`) — parses `choices[i].delta.content` (content), `delta.reasoning_content`/`delta.reasoning` (reasoning), `delta.tool_calls` (indexed id/name/arguments → `tool_started` + `tool_input_delta` events, assembled into `ModelCompletion.tool_calls`), `finish_reason`, and `usage`. Treats `data: [DONE]` and `{"error":...}`.
- `cli_model_catalog_provider` + `model_catalog_provider` — `GET {base}/models`, parse `{"data":[{"id":...}]}`. Entries advertise `has_tool_use`, `has_vision`, `has_file_input`.
- Env config: `FX_OPENAI_BASE_URL` (fallback `OPENAI_BASE_URL`), `FX_OPENAI_API_KEY` (fallback `OPENAI_API_KEY`).

### Edited files (all additive)

- `src/core/config/model_provider.zig` — added `ProviderId.openai_compat`, `parse` aliases (`openai-compat`, `openai_compat`, `openai-compatible`, `custom`, `openai`), `authorizesCredential(.openai_compat, .stored_key)`.
- `src/core/auth/provider_catalog.zig` — catalog entry (`slug: openai-compat`).
- `src/core/gateway/provider_set.zig` — `openai_compat: Bundle = .{}` in `Set` + `select` + `deferredUsageProviders`.
- `src/core/session/generation_usage_provider.zig` — `openai_compat` slot + select.
- `src/core/auth/credentials.zig` — `resolveForProvider(.openai_compat)` loads key from `FX_OPENAI_API_KEY`/`OPENAI_API_KEY` as source `.stored_key`; added `loadOpenAiCompatCredential` + `missing_openai_compat_credential_message`.
- `src/builtins/providers.zig` — wired the bundle (agent_stream + both catalogs; no reviewer, no credits, no deferred usage, no fx_search).
- Exhaustive `ProviderId` switch sites extended with a `openai_compat` case: `app_lifecycle.zig`, `auth_runtime.zig`, `auth_transition.zig`, `app_auth_runtime.zig`, `cli_surface.zig`, `output_contracts.zig`, `settings_store.zig`, `main.zig`.
- `src/main.zig` — added `_ = @import("gateway/openai_compat.zig")` to the test-import block.

## Verification

- `zig build` clean; `zig fmt` clean on all changed files.
- Focused unit tests (in `openai_compat.zig`) pass: request serialization with tools/images, and the SSE reducer (text + reasoning + tool calls + usage).
- Full module graph test run (~2006 tests) showed **zero failures outside** the pre-existing `command_runner` tests that require `FX_TEST_PRODUCT_EXE` (only set by `zig build test`). The edited modules' tests (`model_provider`, `provider_catalog`, `credential_authority`, `generation_usage_provider`, `settings_store`) all pass.
- **Real binary end-to-end via tmux** (see next), plus earlier headless smoke against a local fake server and against the real DeepSeek API.

## tmux end-to-end test (human-like)

Permanent detached tmux session **`fx-e2e`**, project dir `/tmp/opencode/_proj` (contains `app.py`, `util.js`, `README.md`, `pic1.png`, `pic2.png`, `pic3.jpg`, plus the created `greetings.txt`). Driven with real keystrokes and `tmux capture-pane`.

| Step | Outcome |
|------|---------|
| `/model` → picker from live `/v1/models` | Listed all three DeepSeek models; selected `deepseek-v4-flash-vision-exp`; confirmed "Switched to ...". |
| Read `app.py`, summarize `greet`/`add` | Model called `read`+`list`, answered correctly. |
| Rename `misnamed_total` → `total_sum` + docstring | `1 edit` `+2/-1`; verified on disk. |
| Describe `pic3.jpg` | Bug → fixed → detailed description returned. |
| Run `ls -la` | Denied "Review unavailable"; model fell back to read-only `list`. |
| Create `greetings.txt` = `hi from fx` | `1 write`; verified on disk. |

## Bug found and fixed

Image requests failed with `System: request failed: SubscriptionNativeImageUnavailable`.

Root cause: fx's vision router (`src/core/agent/runtime/orchestrator.zig:3013-3026`) only takes the native-images route (where the provider embeds `verified_images`) when the model's resolved capabilities advertise **`supports_vision && supports_file_input`**. The compat catalog marked every model `has_vision=false`, so the request dropped through to the `vision_fallback` branch and, because `openai_compat` has no `vision_fallback` capability, raised `SubscriptionNativeImageUnavailable`.

Fix: in `openai_compat.zig` `fetchModelCatalog`, advertise `has_vision=true` and `has_file_input=true` on entries. Verified fixed in the same tmux session.

## Known limitations / decisions

- **No permission reviewer** (deliberately `null`). This makes auto-mode fail closed for terminal actions that need review (e.g. `ls -la` was denied; the model recovered with a read-only alternative). Reads, edits, and reversible new-file creation still work. Wiring a reviewer means an LLM safety review against your endpoint — a security boundary, intentionally not done silently.
- **Vision advertised for all catalog models** — the `/v1/models` endpoint doesn't report modalities, so vision/file-input is advertised broadly. The user picks the model; a non-vision model may reject images at the server.
- **`usage` reporting** omitted for max compatibility (no `stream_options.include_usage`); billing/deferred usage is null for this provider.
- `CONTROL_VARIABLES.md` at repo root is a pre-existing generated file (untracked, predates this session) and is intentionally **not** part of the feature commit.

## How to use

```bash
export FX_OPENAI_BASE_URL=https://api.deepseek.com   # or OPENAI_BASE_URL
export FX_OPENAI_API_KEY=sk-...                       # or OPENAI_API_KEY (optional for local servers)
```
Select the provider in `~/.fx/settings.json` (`"provider": "openai-compat"`) or `fx provider openai-compat`, then choose a model via `/model`.

## Next steps

1. Decide whether to wire a permission reviewer for the compat provider (enables shell-command execution in auto mode).
2. Optionally refine capability hints per model-id and add `stream_options.include_usage`.
3. Push a draft PR so Full CI runs on the four native runners; address any failures.
4. When ready, follow the release/ready gates in `AGENTS.md`.

## Test environment artifacts (outside repo, /tmp/opencode)

- `launch.sh` — exports the fx env + API key (read from `dskey`), cds to `/tmp/opencode/_proj`, execs `zig-out/bin/fx`.
- `dskey` — DeepSeek API key (mode 600). Never echo it.
- `_fh` — isolated fx profile (`~/.fx/settings.json` with provider `openai-compat`, model `deepseek-v4-flash-vision-exp`).
- `_proj` — test project with files + images + the fx-e2e tmux session.
