---
title: Platforms
description: The three ways to run CyberEther, how they differ, and how to pick one.
order: 240
category: Usage
---

CyberEther runs in three distinct modes: as a **native desktop application**, as a **web application** inside your browser, and as a **remote host** that streams its interface over the network. Your flowgraphs stay identical across all three. The only things that change are where compute runs, which hardware you can reach, and how much setup is involved.

## One Core, Three Shells

Every mode runs the exact same engine, scheduler, blocks, and UI code. A flowgraph saved on one platform opens unchanged on any other. What differs is the thin shell around that core:

```text
  Native     Your Machine [ Compute + GPU Render + Local Files + Attached Hardware ]

  Web        Browser Tab  [ WebAssembly CPU + WebGPU Render + WebUSB + Browser Storage ]

  Remote     Host Machine [ Compute + Hardware + Render ] ── WebRTC ──▶ Any Browser
```

## Side-by-Side Comparison

| Capability | [Native](/docs/native) | [Web](/docs/web) | [Remote](/docs/remote) |
|---|---|---|---|
| **Where compute runs** | Your local machine (all backends) | Your browser tab (CPU via WebAssembly) | The host machine (all native backends) |
| **How UI is drawn** | Metal or Vulkan on your GPU | WebGPU inside the browser | Rendered on the host, streamed as video |
| **How input arrives** | Directly from the window | Directly from the browser tab | Sent from your browser back to the host |
| **Where files live** | Your local disk | Private browser storage | The host's disk |
| **Install required** | Yes | No | On the host only |
| **Setup effort** | Install and launch | Open a link | Install on the host and share a link |
| **Hardware access** | Full (all built-in SDRs, SoapySDR, audio, GPUs) | Built-in USB SDRs via WebUSB (except PlutoSDR and BirriaSDR) | Full, on the host |
| **Python blocks** | Yes (local Python) | Yes (in-browser CPython) | Yes (host Python) |
| **Interface latency** | Lowest | Low | Depends on network quality |
| **Best for** | Daily work, development, deployment | Trying it out, demos, sharing | Headless servers and remote rigs |

## Choosing Your Setup

### Native

Choose [Native](/docs/native) for everyday use, heavy workloads, development, and unattended deployments. It runs directly on your machine with full access to your GPU, every supported compute backend, and attached peripherals like SDRs and audio interfaces.

### Web

Choose [Web](/docs/web) when you want zero setup. The entire framework compiles to WebAssembly and runs locally inside a browser tab with no server behind it. It is perfect for quick experiments, classroom demos, or sharing a flowgraph link with a colleague.

### Remote Streaming

Choose [Remote Streaming](/docs/remote) when your compute or radio hardware lives somewhere else. The native application runs on the host (often headless) and streams its full interactive UI straight to a browser on your laptop, tablet, or phone.

> [!TIP]
> As a simple rule of thumb: build and deploy with **Native**, share quick demos with **Web**, and control remote or headless machines with **Remote Streaming**.
