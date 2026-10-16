---
title: Native
description: Run CyberEther as a native application with direct access to the GPU and local hardware.
order: 40
category: Usage
parent: platforms
---

The native application is the primary way to run CyberEther. It is a plain binary with direct access to the GPU, every compute backend the platform provides, and any hardware attached to the machine, such as SDRs and audio devices. It is the right choice for daily use, development, and unattended deployments.

## Quick Start

After [installing CyberEther](/docs/installation), launch it from a terminal:

```
cyberether
```

This opens the main window with an empty workspace. To load a flowgraph at startup, pass its path as the only positional argument:

```
cyberether flowgraph.yml
```

To run without a window, for example on a server, add `--headless`:

```
cyberether --headless flowgraph.yml
```

## How It Works

CyberEther draws its own interface directly on the GPU and runs the flowgraph on the compute backends the build supports. Nothing sits between the application and the operating system, which is what keeps latency low and lets the same binary scale from a Raspberry Pi to a workstation.

- **Rendering** uses Metal on macOS and Vulkan elsewhere. Pick one explicitly with `--renderer`. When the preferred renderer is unavailable, CyberEther falls back to one that works.
- **Compute** runs on the CPU everywhere, and on CUDA where the hardware and the build allow it. The [installation guide](/docs/installation) covers what each build enables.
- **Headless mode** still runs the full present loop against a windowless viewport, so blocks with visual surfaces keep working and can be captured or streamed with [Remote Streaming](/docs/remote).

## Requirements

- A GPU with Metal or Vulkan support. The [overview](/docs/overview) lists the supported devices.
- For Python blocks, a Python installation on the machine. See [Choosing a Python Runtime](/docs/python-block#choosing-a-python-runtime).

## Command Line

```
cyberether [options] [flowgraph]
cyberether run [options] [flowgraph]
cyberether benchmark [options] [block]
cyberether update [--install]
```

The default command launches the full application. The `benchmark` command runs the benchmark suite, and `update` checks for a newer release. Run `cyberether --help`, or the help of a specific command, for the complete option list with the current values.

Long options with values accept either `--option value` or `--option=value`. Use `--` to treat every following argument as positional, such as for a flowgraph path beginning with a dash.

### Graphics

| Option | Meaning |
|---|---|
| `--renderer <renderer>` | Preferred graphics backend, metal or vulkan. |
| `--device-index <index>` | Zero based Vulkan and CUDA device index. |
| `--headless` | Run without a window. |
| `--size <WxH>` | Viewport size. |
| `--scale <factor>` | Interface scale factor. |
| `--theme <name>` | Interface color theme. |
| `--framerate <fps>` | Target frame rate. |

Vulkan and CUDA enumerate devices independently, so the same index is not guaranteed to identify the same physical GPU in both APIs.

### Runtime

| Option | Meaning |
|---|---|
| `--python-runtime <path>` | Python executable or library for this run. |
| `--dependency-policy <policy>` | Dependency installation policy, prompt, allow, or deny. |
| `--plugin <path>` | Load a plugin. Repeat to load several. |

### Benchmark

The benchmark command accepts `--format markdown`, `--format json`, or `--format csv`. Pass a block type to run only its benchmarks:

```
cyberether benchmark fft --format json
```

The CUDA benchmarks honor the global `--device-index` selection.

### Remote

The `--remote` option and its `--remote-*` settings stream the interface to a browser on another device. They are covered in [Remote Streaming](/docs/remote).

## Settings

CyberEther saves its settings in a `settings.yaml` file:

| System | Location |
|---|---|
| Linux | `~/.config/cyberether/` |
| macOS | `~/Library/Application Support/CyberEther/` |
| Windows | `%APPDATA%\CyberEther\` |

On Linux, `XDG_CONFIG_HOME` replaces `~/.config` when it is set. Most settings are easier to change in the **Settings** window. Command line options override the saved values for one run only and never modify the file.

## Troubleshooting

**The window does not open.** Try the other renderer with `--renderer vulkan` or `--renderer metal`, and check that the GPU driver is up to date.

**The wrong GPU is used.** Pick another device with `--device-index`. On machines with several GPUs, the index can point to a different card in Vulkan and in CUDA.

**Python blocks report an invalid runtime.** Select a working installation in **Settings** under **Runtime**, or pass one with `--python-runtime`.
