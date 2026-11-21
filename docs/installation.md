---
title: Installation
description: How to install CyberEther using pre-built binaries, container images, or the browser edition.
order: 110
category: Getting Started
---

You can run CyberEther from a pre-built release, pull a Docker container, or compile it from source. If you just want to try it without installing anything, open the browser edition at [cyberether.org/web](https://cyberether.org/web).

| Method | Best For | Setup Time |
|---|---|---|
| [Pre-Built Binaries](#pre-built-binaries) | Most desktop users on Linux, macOS, and Windows | Under 1 minute |
| [Container Images](#container-images) | Headless servers, reproducible setups, and GPU clusters | 1 to 2 minutes |
| [Build From Source](/docs/building) | Contributors, custom build flags, or unlisted platforms | 5 to 15 minutes |

## Pre-Built Binaries

Download the latest official release for your operating system from the [CyberEther website](https://cyberether.org).

- Linux releases are provided for both `x86_64` and `aarch64`.
- The `-cuda` AppImages bundle CUDA 12 runtime libraries and require a compatible NVIDIA driver on the host.
- Built-in drivers for RTL-SDR, Airspy, Airspy HF+, HackRF, LimeSDR, bladeRF, PlutoSDR, and BirriaSDR are already compiled in, and any SoapySDR 0.8 or 0.9 installation on your system is detected at runtime. See [Hardware And Devices](/docs/hardware) for platform driver notes such as Windows WinUSB and USB permissions.

Desktop builds can also check for and install updates from the command line using `cyberether update`. See [Native](/docs/native) for command-line flags and configuration paths.

## Container Images

Official Docker images are published to the GitHub Container Registry (`ghcr.io`) for every release:

```bash
# Standard CPU images
docker pull ghcr.io/luigifcruz/cyberether:ubuntu24-x86_64
docker pull ghcr.io/luigifcruz/cyberether:ubuntu24-aarch64

# CUDA-enabled images
docker pull ghcr.io/luigifcruz/cyberether:ubuntu24-x86_64-cuda
docker pull ghcr.io/luigifcruz/cyberether:ubuntu24-aarch64-cuda
```

The CUDA images bundle the CUDA 13 user-space libraries, though your host still needs a compatible NVIDIA driver and the NVIDIA Container Toolkit. Launch them with GPU access enabled:

```bash
docker run --gpus all ghcr.io/luigifcruz/cyberether:ubuntu24-x86_64-cuda
```

The container images do not include ONNX Runtime or the **Infer** block. To use the block, [build from source](/docs/building) with ONNX Runtime installed and `-Dinference=enabled`.

## Building From Source

If your platform is not listed or you want to enable custom Meson build options, cross-compile for WebAssembly or iOS, or run the test suite, follow the [Building From Source](/docs/building) guide.

## Next Steps

With CyberEther installed, jump into the [Quick Start guide](/docs/quick-start) to launch the app and build your first live flowgraph.
