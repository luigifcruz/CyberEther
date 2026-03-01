---
title: Overview
description: A multi-platform GPU-accelerated signal processing framework.
order: 1
category: Getting Started
---

CyberEther is a high-performance GPU-accelerated framework for real-time signal visualization and data processing. It leverages low-level graphics APIs to achieve native performance on any platform.

- 🎨 Graphical support for any device with **Vulkan**, **Metal**, or **WebGPU**.
- 🌐 Installation-free fully-featured web application powered by **WASM** and **WebGPU**.
- 📡 Low-latency **remote interface** for headless servers and edge devices.
- 🔀 Modern **flowgraph editor** for building and running real-time pipelines.
- 🐍 Python API for custom signal visualization via **Superluminal**.

Check out the [installation](/docs/installation) to get started!

## Compatibility 

CyberEther can run in virtually any modern device with a graphics card. The build system will automatically choose between the three graphical backends available (Metal, Vulkan, or WebGPU) depending on the target device.

|    |                Device               |       Graphics        |        Compute        |
|----|-------------------------------------|-----------------------|-----------------------|
| ✅ | Apple Silicon (iPad, iPhone, Mac)   | Metal, Vulkan, WebGPU | CPU                   |
| ✅ | Linux (NVIDIA)                      | Vulkan, WebGPU        | CPU                   |
| ✅ | Linux (AMD, Intel)                  | Vulkan, WebGPU        | CPU                   |
| ✅ | Raspberry Pi (4 or later)           | Vulkan, WebGPU        | CPU                   |
| ✅ | Browser (Chrome)                    | WebGPU                | CPU                   |
| ✅ | Windows (NVIDIA, AMD, Intel)        | WebGPU, Vulkan        | CPU                   |
| ✅ | Android                             | WebGPU, Vulkan        | CPU                   |

The development of compute for CUDA, Vulkan, and WebGPU is currently in progress.
