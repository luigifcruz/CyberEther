---
title: Native
description: Run CyberEther as a native application with direct access to the GPU and local hardware.
order: 241
category: Usage
parent: platforms
---

The native application is the primary way to run CyberEther. Shipped as a standalone binary, it talks directly to your GPU, every compute backend enabled in your build, and local hardware such as SDRs and audio devices. Use it for daily work, development, and unattended deployments.

## Quick Start

After [installing CyberEther](/docs/installation), launch it from a terminal:

```bash
# Open the Start tab with an empty workspace
cyberether

# Launch directly into a specific flowgraph
cyberether flowgraph.yml

# Run without opening a window (ideal for headless servers)
cyberether --headless flowgraph.yml
```

## Under the Hood

Nothing sits between the native binary and your operating system. That direct path keeps latency low and lets the same codebase scale from a Raspberry Pi to a multi-GPU workstation.

- **Graphics rendering** uses Metal on macOS and Vulkan on Linux and Windows. Pass `--renderer` to choose one explicitly. If your requested backend is unavailable, CyberEther automatically falls back to a working one.
- **Compute execution** runs on the CPU everywhere, and on CUDA wherever your hardware and build support it. Check [Building From Source](/docs/building) for build options.
- **Headless execution** still drives the full presentation loop against a windowless viewport. Visual blocks continue rendering surfaces that can be captured or streamed over [Remote Streaming](/docs/remote).

## Requirements

- A GPU supporting Metal or Vulkan (see [Building From Source](/docs/building) for platform requirements).
- A local Python installation if you use Python blocks (see [Choosing a Python Runtime](/docs/python-runtime#choosing-a-python-runtime)).

## Command-Line Reference

```bash
cyberether [options] [flowgraph]
cyberether run [options] [flowgraph]
cyberether benchmark [options] [block]
cyberether update [--install]
```

Running `cyberether` (or `cyberether run`) opens the application. Use `benchmark` to run the performance suite and `update` to check for or install newer releases. Run `cyberether --help`, or pass `--help` to any subcommand, for the complete list of flags and defaults.

The flowgraph argument also accepts a gallery link. A link such as `cyberether://flowgraph/first-steps` opens that example from the [gallery](https://cyberether.org/gallery/flowgraphs), and a link of the form `cyberether://plugin/<id>` opens the details of the plugin with that ID, where you can choose to install it. Either kind can name the owner first, as in `cyberether://flowgraph/cyberether/first-steps`, and a link without an owner refers to the official gallery. A link must be the only argument, and when CyberEther is already running it opens in that window instead of starting a second one.

Long options that take a value accept both `--option value` and `--option=value`. Insert `--` to treat every argument after it as positional, which is useful when a flowgraph filename begins with a dash.

### Global Options

| Option | Purpose |
|---|---|
| `-h`, `--help` | Show usage help for CyberEther or a specific subcommand. |
| `-v`, `-vv` | Increase log verbosity (`-v` for debug logs, `-vv` for trace logs). |
| `-V`, `--version` | Print the installed CyberEther version and exit. |

### Graphics Options

| Option | Purpose |
|---|---|
| `--renderer <renderer>` | Select the preferred graphics backend (`metal` or `vulkan`). |
| `--device-index <index>` | Zero-based Vulkan device index. |
| `--headless` | Run without creating a desktop window. |
| `--size <WxH>` | Set the viewport width and height. |
| `--scale <factor>` | Set the UI scaling factor. |
| `--theme <name>` | Choose the interface color theme. |
| `--framerate <fps>` | Set the target presentation frame rate. |

> [!NOTE]
> The CUDA backend follows the graphics card that Vulkan uses by matching the device UUID. When no CUDA device matches, it uses the first CUDA device. The startup log lists every Vulkan GPU with its index.

### Runtime Options

| Option | Purpose |
|---|---|
| `--python-runtime <path>` | Override the Python executable or library for this session. |
| `--dependency-policy <policy>` | Set the Python dependency installation policy (`prompt`, `allow`, or `deny`). |
| `--plugin <path>` | Load a `.cep` plugin bundle for this run. Repeat the flag to load multiple plugins. |

### Benchmarking

The `benchmark` subcommand supports `--format markdown`, `--format json`, and `--format csv`. Pass a block type to benchmark a single block:

```bash
cyberether benchmark fft --format json
```

Benchmarks run CUDA on the graphics device saved in **Settings** when a CUDA device matches it, and on the first CUDA device otherwise. The `--device-index` flag only selects the Vulkan device.

### Updating CyberEther

Official desktop packages built with Velopack support checking for and installing updates from the terminal:

```bash
# Check whether a newer release is available
cyberether update

# Download and install the update
cyberether update --install
```

### Remote Streaming

Pass `--remote` along with any `--remote-*` flags to stream the interface to another device. See the [Remote Streaming guide](/docs/remote) for every available flag and configuration recipe.

## Configuration File

CyberEther ships with sensible defaults for every setting and saves changes to `settings.yaml` only when you edit them in **Settings** (or create the file manually). To inspect all command-line defaults for your installed build, run `cyberether --help` or `cyberether <subcommand> --help`.

| Operating System | Directory |
|---|---|
| Linux | `~/.config/cyberether/` |
| macOS | `~/Library/Application Support/CyberEther/` |
| Windows | `%APPDATA%\CyberEther\` |

On Linux, setting the `XDG_CONFIG_HOME` environment variable overrides `~/.config`. Most preferences are easiest to change in the **Settings** tab. Command-line flags override saved settings for that run only and never modify `settings.yaml`.

The `settings.yaml` file organizes persisted settings into top-level sections:

- **`graphics`:** Preferred graphics backend and device, window size, UI scale, and frame rate. Session flags such as `--device-index` and `--headless` are not persisted.
- **`remote`:** Access approval mode, video codec and encoder, broker address, port, TLS paths, and media transport settings.
- **`runtime`:** Selected Python runtime path and PEP 723 dependency policy.
- **`interface`:** Active theme, background particles, tips, touchpad scroll speed, and recent or favorite flowgraphs.
- **`maps`:** High-resolution map toggle and map rendering preferences.
- **`developer`:** Log level, compute timing overlay, and render stats flags.
- **`intelligence`:** Optional Assistant providers, model selection, voice settings, and the built-in MCP server (see [Intelligence](/docs/intelligence)).

## Troubleshooting

| Symptom | What to Try |
|---|---|
| **The window fails to open** | Switch renderers with `--renderer vulkan` or `--renderer metal`, and confirm your GPU drivers are up to date. |
| **The wrong GPU is active** | With the Vulkan renderer, pick the card under **Settings** > **General** > **Graphics Device** and restart CyberEther, or pass `--device-index` for a single run. Metal always uses the system default device. A saved GPU that is not connected falls back to the automatic choice without losing the setting. |
| **Python blocks report an invalid runtime** | Select a working interpreter under **Settings** > **Runtime**, or pass one explicitly with `--python-runtime`. |
