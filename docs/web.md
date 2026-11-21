---
title: Web
description: Run CyberEther inside a browser tab with nothing to install.
order: 242
category: Usage
parent: platforms
---

The web edition runs the full CyberEther engine directly inside your browser tab using WebAssembly and WebGPU. It is not a remote frontend talking to a server. Everything, including the flowgraph editor, scheduler, and DSP blocks, executes locally on your machine with zero installation.

It is the fastest way to explore CyberEther, run a live demo, or share a flowgraph with someone else.

## Quick Start

Open [cyberether.org/web](https://cyberether.org/web) in a WebGPU-enabled browser such as a recent version of Chrome. You get the same interface and built-in example gallery as the desktop application.

## How the Browser Build Works

The web edition compiles from the exact same codebase as the native app, swapping only the platform integration layer:

- **Rendering** targets WebGPU, giving the UI direct access to your GPU through the browser.
- **Compute** runs each block's CPU implementation compiled to WebAssembly. WebGPU compute backends are not yet available.
- **Hardware radios** connect over WebUSB for supported built-in drivers (`RTL-SDR`, `Airspy`, `Airspy HF+`, `HackRF`, `LimeSDR`, and `bladeRF`) once you grant the page access to the device. See [Hardware And Devices](/docs/hardware) for details.
- **Python blocks** run inside a WebAssembly build of CPython bundled by the page rather than a local Python installation.
- **The main loop** is driven by the browser's animation loop, since tabs cannot own a blocking main loop. Modules that require the browser's main thread are automatically proxied there for creation and destruction.
- **Files and settings** live inside private browser storage instead of your local filesystem.

## Requirements

- **A WebGPU-capable browser.** Chrome and other Chromium-based browsers enable WebGPU and WebUSB by default on most desktops. Other browsers may require turning WebGPU on in their settings or may not support WebUSB.
- **A reasonably fast CPU**, since all signal processing currently executes on the CPU in WebAssembly.

## Files and Storage

Your flowgraphs and settings are saved in the browser's **Origin Private File System**, a sandboxed storage area tied to the website. Files persist across visits on the same browser and device, and opening or saving files uses the web page's file dialogs.

> [!WARNING]
> Clearing browser site data deletes everything stored in the Origin Private File System, including saved flowgraphs and settings. Always keep copies of important flowgraphs on your local disk.

## Limitations Compared to Native

- **Hardware access** is limited to browser APIs. Built-in USB radios work over WebUSB except for PlutoSDR and BirriaSDR, and SoapySDR device modules are only available on desktop.
- **Compute throughput** is CPU-only, making very heavy flowgraphs slower than on native builds.
- **Remote Streaming** is unavailable because it depends on native video encoders and networking stacks.

Whenever you need PlutoSDR, BirriaSDR, SoapySDR, GPU compute, or remote streaming, switch to the [Native application](/docs/native) or host it on another machine with [Remote Streaming](/docs/remote).

## Self-Hosting Your Own Build

The browser build has no install target. Compiling it with the [Building From Source](/docs/building) guide outputs `cyberether.js` and `cyberether.wasm` in `build-wasm/`.

To host those files yourself, make sure your web server:

1. Serves over a WebGPU-capable context (HTTPS or `localhost`).
2. Sends cross-origin isolation headers (`Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy`), which browsers require for CyberEther's WebAssembly threads.
3. Serves `.wasm` files with the `application/wasm` content type.

> [!TIP]
> Want to test a custom build without configuring a web server? Open [cyberether.org/web](https://cyberether.org/web), which lets you load your own local `cyberether.js` and `cyberether.wasm` files directly.

## Troubleshooting

| Symptom | Likely Cause and Fix |
|---|---|
| **The app fails to load** | Verify that your browser supports WebGPU and has it enabled. Updating the browser often helps, and some browsers keep WebGPU disabled on older GPUs. |
| **A USB radio does not appear** | Confirm your browser supports WebUSB (Chrome or Chromium-based browsers) and that you granted USB device access when prompted. PlutoSDR and BirriaSDR require the native app. |
| **Saved flowgraphs vanished** | Site data was cleared, or you opened the page in a private or incognito window whose storage is wiped when the window closes. |
| **A self-hosted build refuses to start** | Check that your server sends the cross-origin isolation headers and serves `cyberether.wasm` with the proper MIME type. |
