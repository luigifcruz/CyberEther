---
title: Web
description: Run CyberEther inside a browser tab with nothing to install.
order: 41
category: Usage
---

The web version is the same CyberEther compiled to WebAssembly and drawn with WebGPU. It is not a web frontend to a native process. The flowgraph, the scheduler, and the signal processing all run inside the browser tab, so there is nothing to install and nothing running on a server. It is the quickest way to try CyberEther, give a demo, or share a flowgraph with someone.

## Quick Start

Open [cyberether.org/web](https://cyberether.org/web) in a browser with WebGPU enabled, such as a recent version of Chrome. The application loads in the tab with the same interface and examples as the native version.

## How It Works

The browser build shares its code with the native application. Only the layer that talks to the system changes.

- **Rendering** uses WebGPU, the browser's interface to the GPU.
- **Compute** runs the CPU implementation of each block, compiled to WebAssembly. No blocks target WebGPU compute yet.
- **The main loop** is driven by the browser's animation loop, since a browser tab cannot own its main loop. Modules that must run on the main thread are moved there for creation and destruction.
- **Files and settings** live in browser storage, private to the site, instead of on your disk.

## Requirements

- A browser with WebGPU support. Chrome and other Chromium based browsers support it on most desktops. Other browsers may need it turned on in their settings.
- A reasonably fast CPU, since all compute runs on it.

## Files and Storage

Flowgraphs and settings are stored in the Origin Private File System, a storage area the browser keeps for each website. They persist across visits on the same browser and device. Opening and saving files go through file dialogs provided by the web page.

Clearing the browser data for the site deletes everything stored there, including saved flowgraphs and settings. Keep copies of important flowgraphs on your disk.

## Limitations

- **Python blocks** are unavailable, since they need a Python installation that does not exist inside the browser sandbox.
- **Hardware access** is limited to what browser APIs expose, so sources that need native drivers, such as most SDRs, are out of reach.
- **Compute** runs on the CPU only, so large flowgraphs run slower than on the native version.
- **Remote Streaming** is not available, since it relies on native encoders and networking.

For any of these, use the [native application](/docs/native), or run it on another machine and stream it with [Remote Streaming](/docs/remote).

## Self-Hosting

The browser build has no install target. Building it, as described in the [installation guide](/docs/installation), produces `cyberether.js` and `cyberether.wasm` in the build directory. Serve them from a website that supports WebGPU and has cross-origin isolation enabled, which the browser requires for the threads CyberEther uses.

The hosted app at [cyberether.org/web](https://cyberether.org/web) can also load custom builds, which is the easiest way to test your own files.

## Troubleshooting

**The application does not load.** Check that the browser supports WebGPU and has it turned on. Updating the browser often helps, and some browsers keep WebGPU disabled on certain GPUs.

**Saved flowgraphs disappeared.** The browser data for the site was cleared, or the page was opened in a private window, which uses separate storage that is deleted when the window closes.

**A self-hosted build fails to start.** Check that the server sends the cross-origin isolation headers and serves the WebAssembly file with the correct content type.
