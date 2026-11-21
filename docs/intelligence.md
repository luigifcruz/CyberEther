---
title: Intelligence
description: Use the built-in Assistant, configure providers and voice, and connect external tools via the local MCP server.
order: 260
category: Usage
---

CyberEther includes an optional **Assistant** panel and a built-in **MCP Server** that can inspect open flowgraphs, read block documentation, diagnose errors, and apply structured edits. Both features are disabled by default and are available in native desktop and server builds (browser builds do not support Intelligence).

## Enabling Intelligence

Open **Settings** > **Intelligence** and turn on **Enable Intelligence** under the **Assistant** group. Turning Intelligence off hides every AI feature and disconnects all configured providers.

Once enabled, press **Cmd/Ctrl + L** to toggle the **Assistant** tile on the right side of the workbench.

## Providers and API Keys

Under **Settings** > **Intelligence** > **Providers**, CyberEther includes five provider presets, all using OpenAI-compatible HTTP endpoints:

| Provider Preset | Request API | Notes |
|---|---|---|
| **OpenAI** | Responses | Supports chat, reasoning models, and live Voice mode. |
| **OpenRouter** | Chat Completions | Aggregates third-party models. Its model catalog lists models even without a valid key, so a passing catalog test does not guarantee chat access. |
| **OpenCode Go** | Chat Completions | Hosted OpenAI-compatible endpoint. |
| **Fireworks AI** | Chat Completions | Hosted OpenAI-compatible endpoint. |
| **Custom** | Chat Completions or Responses | Point to any local or remote OpenAI-compatible server (such as Ollama, vLLM, or a proxy). Blank API keys are allowed for unauthenticated local servers. |

Before saving a provider, CyberEther requires a passing connection test.

### Managing Credentials

Every provider supports two mutually exclusive API key sources:

- **Keychain:** Stores the secret in your operating system's native secret store (`CyberEther Intelligence`). Only a secret identifier (`apiKeyId`) is written to `settings.yaml`, never the raw key.
- **Environment:** Reads the key on each request from a named environment variable (`apiKeyEnv`) without storing the secret on disk. A configured environment variable that is missing or unset does **not** fall back to Keychain.

### Models and Reasoning Effort

After adding a provider, open the **Models** group and click **Add** to open the **Add Model** picker and choose models from that provider's catalog. CyberEther automatically pulls metadata such as context length, output limits, optional pricing, and supported reasoning levels. When a model supports reasoning effort, that setting is stored per model rather than globally.

## Voice Mode

When CyberEther is compiled with live voice support, the **Voice** group lets you speak directly with the Assistant. Voice mode currently requires a configured **OpenAI** backend model and API key.

Voice uses a two-model architecture rather than a separate speech-to-text and text-to-speech pipeline:

- **Voice Model:** A live audio model that listens and speaks over a real-time connection.
- **Backend Model:** The configured model that performs deeper reasoning and executes flowgraph tool calls.
- **Voice:** Picks the spoken voice preset (such as `marin`, `cedar`, `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`, `shimmer`, or `verse`). This selects the speaker voice rather than a system audio device.

## Using the Assistant

Open the **Assistant** tile (**Cmd/Ctrl + L**) to ask questions or modify your active pipeline.

- **Mid-turn steering:** Sending a message while the Assistant is streaming a text reply queues it so it can steer the next model step (or run as the next turn). You can withdraw a queued message before it is absorbed. Message queueing is unavailable during active voice sessions.
- **Turn-scoped focus:** When you start a chat turn or voice session, the Assistant binds to the flowgraph active at that moment. Clicking another tab mid-turn does not redirect running tools unless the Assistant explicitly switches tabs using its flowgraph tools.

### Available Tool Groups

Both the built-in Assistant and the MCP server expose the same toolset:

- **Workspace and flowgraphs:** List and open favorite or recent files (`list_flowgraphs` and `open_flowgraph`, restricted to exact paths returned by `list_flowgraphs`), list or focus open tabs (`list_open_flowgraphs` and `focus_flowgraph`), and inspect or arrange dock tiles (`inspect_tiles` and `arrange_tiles`).
- **Catalog and examples:** Search built-in and plugin blocks (`search_blocks`), read per-block docs (`get_block_docs`), and browse or open gallery examples (`list_examples`, `get_example`, `open_example`).
- **Inspection and diagnostics:** Inspect graph structure (`inspect_flowgraph`), block configs and metrics (`inspect_block`, `read_block_text`), CPU tensor values (`inspect_tensor`), environment keys (`inspect_environment`), rendered surface snapshots (`inspect_block_view`, when the renderer supports capture), valid wiring options (`find_connection_options`), validation state (`validate_flowgraph`), and logs (`read_log`).
- **Editing:** Apply structured edits with `apply_flowgraph_patch` (creating, reconfiguring, deleting, connecting, and laying out blocks, or creating a new flowgraph). Patches run sequentially rather than atomically, so if a later operation fails, earlier changes remain applied. Inspect the graph before retrying.
- **Documentation:** Search and read this bundled documentation set (`search_docs` and `read_docs`).

## Built-In MCP Server

CyberEther can expose its flowgraph tools to external MCP-compatible clients (such as coding agents or desktop assistants) over local HTTP.

### Setup

1. Under **Settings** > **Intelligence**, turn on both **Enable Intelligence** and **Enable MCP Server**. You do not need a configured model API key just to serve MCP tools.
2. Check the status badge (`Listening`, `Failed`, or `Stopped`) to confirm the server started.
3. Point your HTTP MCP client at `http://127.0.0.1:7373/mcp` (or your configured address and port).

### Protocol Details

- **Transport:** Streamable HTTP JSON-RPC 2.0 using **POST** requests to `/mcp` (other paths return `404`, and non-POST methods return `405`). Standard input/output (`stdio`) and Server-Sent Events (`SSE`) endpoints are not used.
- **Methods and versions:** Implements `initialize`, `ping`, `tools/list`, and `tools/call` across MCP protocol versions `2025-11-25` and `2025-06-18`, with read-only hints attached to inspection tools.
- **Active target:** Unlike the in-app Assistant, every MCP tool call acts on whichever flowgraph is **currently focused** in the CyberEther window when the call runs.

### Security

The MCP server listens on loopback (`127.0.0.1`) by default, though the bind address and port are configurable (for example `0.0.0.0` on a private network). Keep three safeguards in mind:

- **No authentication:** Any client that can reach the socket can call tools and edit your focused flowgraph.
- **Origin filtering only:** The server rejects browser requests whose `Origin` header is not loopback, which does not protect against non-browser clients on a shared network.
- **Keep it private:** Leave the address at `127.0.0.1` for local use, and never expose the MCP server directly to the internet.
