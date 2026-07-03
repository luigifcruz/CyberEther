---
title: Platforms
description: The three ways to run CyberEther, and what changes between them.
order: 39
category: Usage
---

The same CyberEther runs in three shapes: as a native application on your machine, as a WebAssembly build inside the browser, and as a remote instance that streams its interface to you from somewhere else. The flowgraphs are identical across all three. What changes is where compute happens, which backends exist, and a handful of quirks worth knowing before you pick one.

## Native

The primary target. CyberEther runs as a plain native binary with direct access to the GPU and every backend the platform provides:

```
cyberether [command] [options] [flowgraph]
```

The default command runs the full application and `benchmark` runs the benchmark suite. The graphics options select the render device (`--device metal` or `--device vulkan`), the window geometry (`--size`, `--scale`, `--framerate`), and `--headless` runs without a window entirely. Passing a flowgraph file loads it at startup, which combined with `--headless` is the deployment shape for unattended machines.

Quirks:

- Backend availability follows the OS: Metal on macOS, Vulkan elsewhere, with CUDA compute where the hardware and build allow. The [installation guide](/docs/installation) covers what each build enables.
- Headless mode still runs the full present loop against a windowless viewport, so blocks with visual surfaces keep working and can be captured or streamed.
- The Python runtime binds to a Python installation on the machine, selected in the settings. See [Choosing a Python Runtime](/docs/python-block#choosing-a-python-runtime).

## Web

The browser build is the same application compiled to WebAssembly, running on WebGPU. It is not a web frontend to a native process and it involves no JavaScript in the core: the flowgraph, the scheduler, and the DSP all execute inside the browser tab.

Quirks:

- Rendering runs on WebGPU, so the browser needs WebGPU support enabled. Compute runs the blocks' CPU implementations compiled to WebAssembly, since no modules target WebGPU compute yet.
- The browser has no main-loop ownership to give away, so the application runs from the browser's animation loop, and modules that must touch the main thread declare the `BROWSER_MAIN_THREAD` taint and are proxied there for creation and destruction.
- Python blocks are unavailable, since they bind to a system Python installation that does not exist inside the browser sandbox.
- Files live in a browser-managed filesystem rather than your disk, so flowgraphs are saved and loaded through the browser's storage.
- Device I/O is limited to what browser APIs expose, so hardware sources that need native drivers are out of reach.

## Remote

A remote instance is a native CyberEther, usually headless on a server, that streams its interface to you and takes interaction back, so the full application runs at the remote machine's compute capacity. This platform will be available soon.

## Picking One

| | Native | Web |
|---|---|---|
| Compute | Local, all backends | In-browser, CPU via WebAssembly |
| Hardware access | Full | Browser APIs only |
| Python blocks | Yes | No |
| Install required | Yes | No |
| Best for | Daily use, development, deployment | Trying it, demos, sharing |

The rule of thumb: develop and deploy native, share via web, and stream from big hardware once remote lands.
