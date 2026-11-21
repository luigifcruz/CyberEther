---
title: Overview
description: A multi-platform GPU-accelerated signal processing framework.
order: 100
category: Getting Started
---

CyberEther is a real-time signal processing and visualization framework built from scratch on top of Vulkan, Metal, and WebGPU. Rather than wrapping a signal engine inside a heavyweight desktop toolkit, CyberEther draws its entire interface and plots directly on the GPU while running your processing pipeline on a dedicated compute thread.

That architecture gives you one consistent tool that feels crisp on a laptop, scales up to a workstation GPU, runs headless on a Raspberry Pi or server, and compiles to the exact same application inside a browser tab.

## Where to Start

| If You Want To... | Start Here |
|---|---|
| Try CyberEther right now in Chrome with no install | [Web Edition](/docs/web) |
| Download a binary or pull a container image | [Installation](/docs/installation) |
| Build your first live spectrum analyzer in five minutes | [Quick Start](/docs/quick-start) |
| Compare Native, Web, and Remote Streaming setups | [Platforms](/docs/platforms) |
| Browse every built-in block by domain | [Blocks Catalog](/docs/blocks) |
| Write custom NumPy or CuPy processing inside a graph | [Python Block](/docs/python-block) |
| Compile from source, run tests, or write C++ blocks | [Building From Source](/docs/building) and [Architecture](/docs/architecture) |

## What Makes CyberEther Different

### Live Flowgraphs That Respond as You Build

You build pipelines visually by placing blocks on a canvas, wiring their ports together, and adjusting parameters inline. Because the graph runs continuously while you edit, spectra, waterfalls, and constellations update the moment you connect a wire or tune a field. Flowgraphs save to plain YAML files and open identically across Linux, macOS, Windows, and the web.

### Fixed-Shape, Zero-Copy Tensors

Every sample moving through CyberEther travels inside a typed multidimensional tensor whose shape is settled when the graph is built. Allocating buffers up front and passing them by reference eliminates hidden copies between blocks, across devices, and into visualization surfaces, keeping latency low and behavior predictable.

### Ready for Real Signals Out of the Box

CyberEther ships with over sixty built-in blocks covering tensor math, FFTs, filters, demodulators, audio playback, network streaming, 2D and 3D plots, and ONNX machine learning inference. Drivers for eight popular USB radios are compiled directly into the binary, and any SoapySDR 0.8 or 0.9 installation on your system is picked up automatically at runtime. See [Hardware And Devices](/docs/hardware) for the full compatibility list.

### First-Class Python Integration

When you want to prototype an idea quickly or pull in NumPy, SciPy, or CuPy, drop a **Python** block straight into your flowgraph. Your script receives zero-copy views over CPU and CUDA tensors alongside live metadata and metrics. For standalone scripts that need custom real-time signal plotting, CyberEther also exposes a C++ and Python plotting API via [Superluminal](/docs/superluminal).

### Three Ways to Run

The same core powers three deployment modes:

- **Native:** Direct access to your GPU, compute backends, local files, and attached hardware on desktop and embedded systems.
- **Web:** The full C++ application compiled to WebAssembly and rendered with WebGPU inside a browser tab, with nothing to install and WebUSB support for most built-in SDR drivers.
- **Remote:** A self-hosted native instance (often headless) that streams its interactive interface over encrypted WebRTC to any browser on your laptop, tablet, or phone with no cloud service in between.

## How a Pipeline Comes Together

```text
  Sources                 Processing                 Outputs & Views
  ┌──────────────────┐    ┌─────────────────────┐    ┌───────────────────┐
  │ SDR / SDR Sweep  │    │ FFT, Filters, AGC   │    │ Lineplot / Audio  │
  │ File / Network   │───▶│ Demodulators, Math  │───▶│ Waterfall / 3D    │
  │ Signal Generator │    │ Python / ONNX / C++ │    │ Constellation     │
  └──────────────────┘    └─────────────────────┘    └───────────────────┘
```

1. **Sources** ingest live samples from radios, files, network sockets, or synthetic waveform generators into fixed-size tensors.
2. **Processing blocks** transform, filter, demodulate, or analyze those tensors on the compute thread, extending easily through Python scripts or native `.cep` plugins.
3. **Visualization blocks** render live GPU surfaces right inside each node or in detached dock tiles at display rate without blocking compute.
