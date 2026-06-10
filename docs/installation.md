---
title: Installation
description: How to install CyberEther on your system.
order: 2
category: Getting Started
---

CyberEther can be installed through pre-built binaries, container images, or by building from source. Pick the option that matches your platform and workflow.

## Pre-built Binaries

The easiest way to install CyberEther is to download a pre-built binary from the [CyberEther website](https://cyberether.org).

If your platform is not listed or you want to customize the build, follow the [Build From Source](#build-from-source) instructions below.

## Container Images

Docker images are also published to the GitHub Container Registry (`ghcr.io`) for each release:

```bash
docker pull ghcr.io/luigifcruz/cyberether:ubuntu24-x86_64
docker pull ghcr.io/luigifcruz/cyberether:ubuntu24-aarch64
```

## Build From Source

CyberEther requires a C++20 compiler (GCC 11+ or Clang 14+), [Meson](https://mesonbuild.com) 1.11+, and [Ninja](https://ninja-build.org). Most third-party libraries are bundled as Meson subprojects, so the packages below focus on the platform toolchain, graphics headers or SDKs, shader tools, and Python modules used during resource generation.

If your system blocks global `pip` installs, use a Python virtual environment or equivalent distribution packages.

### Dependencies
<!-- [NEW DEPENDENCY HOOK] -->

<details>
<summary>Linux (Arch Linux)</summary>

#### Linux (Arch Linux)
Core build tools, graphics headers, and desktop integration.
```bash
pacman -S git base-devel pkgconf python python-pip glslang \
          vulkan-headers vulkan-icd-loader zenity
```

Display server headers used by the bundled GLFW build.
```bash
pacman -S libx11 libxcursor libxi libxinerama libxrandr \
          libxkbcommon wayland wayland-protocols systemd
```

Python build modules.
```bash
python -m pip install meson ninja numpy mapbox_earcut pyyaml
```

#### Optional

Remote streaming support uses bundled GStreamer sources, but still needs a few generator tools.
```bash
pacman -S bison flex nasm
```

No system GLFW, SoapySDR, or GStreamer packages are required for the normal source build.

</details>

<details>
<summary>Linux (Ubuntu 22.04)</summary>

#### Linux (Ubuntu 22.04)
Core build tools, graphics headers, and desktop integration.
```bash
apt install git build-essential pkg-config python3 python3-pip glslang-tools \
            libvulkan-dev mesa-vulkan-drivers zenity
```

Display server headers used by the bundled GLFW build.
```bash
apt install libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev \
            libxkbcommon-dev libwayland-dev wayland-protocols libudev-dev
```

Python build modules.
```bash
python3 -m pip install meson ninja numpy mapbox_earcut pyyaml
```

#### Optional

Remote streaming support uses bundled GStreamer sources, but still needs a few generator tools.
```bash
apt install bison flex nasm
```

No system GLFW, SoapySDR, or GStreamer packages are required for the normal source build.

</details>

<details>
<summary>macOS 13+ (Apple Silicon)</summary>

#### macOS 13+ (Apple Silicon)
This assumes [Homebrew](https://brew.sh) and Xcode Command Line Tools are installed. Older versions of macOS might work but installing a newer Clang compiler (14+) will be necessary. Metal on Intel-based Macs is not supported by CyberEther. As a workaround, make sure to install the optional Vulkan dependencies listed below.

Core build and shader tools.
```bash
brew install pkg-config glslang spirv-cross
```

Python build modules.
```bash
python3 -m pip install meson ninja numpy mapbox_earcut pyyaml
```

#### Optional

Remote streaming support uses bundled GStreamer sources, but still needs NASM.
```bash
brew install nasm
```

No system GLFW, SoapySDR, or GStreamer packages are required for the normal source build.

Vulkan backend (required for Intel-based Macs).
```bash
brew install molten-vk vulkan-tools vulkan-headers
```

</details>

<details>
<summary>Windows</summary>

#### Windows

Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with the C++ workload, [Python 3](https://www.python.org), and the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home). If you use Chocolatey, install the remaining build helper with:
```powershell
choco install pkgconfiglite -y
```

Python build modules.
```powershell
python -m pip install meson ninja numpy mapbox_earcut pyyaml
```

#### Optional

Remote streaming support uses bundled GStreamer sources, but still needs NASM and WinFlexBison.
```powershell
choco install nasm winflexbison3 -y
```

</details>

<details>
<summary>Browser (Chrome)</summary>

#### Browser (Chrome)

All CyberEther runtime dependencies for the browser are included in the repository. Make sure [Emscripten](https://emscripten.org/docs/getting_started/downloads.html), [Rust Cargo](https://www.rust-lang.org/tools/install), Python 3, `pkg-config`, `glslangValidator`, and `spirv-cross` are available.

On Debian or Ubuntu, the CI-equivalent packages are:
```bash
apt install git build-essential pkg-config python3-pip glslang-tools spirv-cross cargo
python3 -m pip install meson ninja numpy mapbox_earcut pyyaml
```

</details>

<details>
<summary>iOS (iPhone & iPad)</summary>

#### iOS (iPhone & iPad)

Build iOS from macOS with the latest [Xcode](https://developer.apple.com/xcode/) installed. You also need the same Homebrew shader tools and Python build modules used by the macOS build.
```bash
brew install pkg-config glslang spirv-cross
python3 -m pip install meson ninja numpy mapbox_earcut pyyaml
```

</details>

### Clone

Clone the repository from GitHub.
```bash
git clone --recursive https://github.com/luigifcruz/CyberEther.git
cd CyberEther
```

### Build & Install

<details>
<summary>Linux or macOS</summary>

#### Linux or macOS

Create a debug-optimized build directory, compile CyberEther, and install the resulting binary and libraries.
```bash
meson setup -Dbuildtype=debugoptimized build
meson compile -C build
meson install -C build
```

To compile the optional remote streaming support, add `-Dremote=enabled` to the setup command.

After installation, run `cyberether --help` for usage instructions.

</details>

<details>
<summary>Windows</summary>

#### Windows

Configure, build, and install from a Visual Studio Developer PowerShell or any shell where the Visual Studio toolchain is available.
```powershell
meson setup --vsenv -Dbuildtype=debugoptimized build
meson compile -C build
meson install -C build
```

To compile the optional remote streaming support, add `-Dremote=enabled` to the setup command.

After installation, run `cyberether --help` for usage instructions.

</details>

<details>
<summary>Browser (Chrome)</summary>

#### Browser (Chrome)

Build the project with cross-compilation to WebAssembly.
```bash
meson setup --cross-file meson/crosscompile/emscripten.ini \
            -Dbuildtype=debugoptimized \
            build-wasm
meson compile -C build-wasm
```

The browser build does not have an install target. The distributable outputs are generated in `build-wasm/`.
```text
build-wasm/cyberether.js
build-wasm/cyberether.wasm
build-wasm/cyberether.js.symbols
build-wasm/cyberether.wasm.map
```

Serve those files from a WebGPU-capable website with cross-origin isolation enabled, which is required by Emscripten pthreads. The hosted browser app at [cyberether.org/web](https://cyberether.org/web) can also load custom builds, so you can test your own `cyberether.js` and `cyberether.wasm` files. A fully offline browser version is on the roadmap.

</details>

<details>
<summary>iOS (iPhone & iPad)</summary>

#### iOS (iPhone & iPad)

Build the project with cross-compilation to iOS and install binaries in the Xcode project.
```bash
meson setup --cross-file meson/crosscompile/ios.ini \
            --prefix "$(pwd)/apps/ios/CyberEtherMobile/Library" \
            -Dbuildtype=debugoptimized \
            -Dtests=false \
            -Dexamples=false \
            build-ios
meson compile -C build-ios
meson install -C build-ios
```

After the build is complete, open the Xcode project in `apps/ios/` and run it on your device.

</details>
