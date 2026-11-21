---
title: Building From Source
description: Install dependencies, configure Meson options, cross-compile for WebAssembly or iOS, and run the test suite.
order: 300
category: Development
---

Building CyberEther from source requires a C++20 compiler (GCC 11+ or Clang 14+), [Meson](https://mesonbuild.com) 1.11+, [Ninja](https://ninja-build.org), and Python 3. Because CyberEther bundles most third-party libraries as Meson subprojects, you only need your platform toolchain, graphics headers or SDKs, shader tools, and a handful of Python build packages.

> [!TIP]
> If your operating system blocks global `pip` installs, create a Python virtual environment or install the equivalent packages from your distribution's package manager.

## Dependencies
<!-- [NEW DEPENDENCY HOOK] -->

> [!NOTE]
> Standard source builds do not need system GLFW, SoapySDR, or GStreamer packages. Drivers for RTL-SDR, Airspy, Airspy HF+, HackRF, LimeSDR, bladeRF, PlutoSDR, and BirriaSDR are built in. Both the SDR and SDR Sweep blocks also detect and use a SoapySDR 0.8 or 0.9 installation and its device modules whenever one is present at runtime.

Expand your target platform below to install the required packages.

<details>
<summary>Linux (Arch Linux)</summary>

### Required Packages

Core build tools, graphics headers, and desktop integration:

```bash
pacman -S git base-devel pkgconf python python-pip glslang \
          vulkan-headers vulkan-icd-loader zenity
```

Display server headers required by the bundled GLFW build:

```bash
pacman -S libx11 libxcursor libxi libxinerama libxrandr \
          libxkbcommon wayland wayland-protocols systemd
```

Python build modules:

```bash
python -m pip install meson ninja numpy
```

### Optional Features

Remote streaming uses bundled GStreamer sources and requires these generator tools:

```bash
pacman -S bison flex nasm
```

Machine learning inference requires ONNX Runtime:

```bash
python -m pip install onnxruntime        # CPU
python -m pip install onnxruntime-gpu    # CPU + CUDA
```

</details>

<details>
<summary>Linux (Ubuntu 22.04)</summary>

### Required Packages

Core build tools, graphics headers, and desktop integration:

```bash
apt install git build-essential pkg-config python3 python3-pip glslang-tools \
            libvulkan-dev mesa-vulkan-drivers zenity
```

Display server headers required by the bundled GLFW build:

```bash
apt install libx11-dev libxcursor-dev libxi-dev libxinerama-dev libxrandr-dev \
            libxkbcommon-dev libwayland-dev wayland-protocols libudev-dev
```

Python build modules:

```bash
python3 -m pip install meson ninja numpy
```

### Optional Features

Remote streaming uses bundled GStreamer sources and requires these generator tools:

```bash
apt install bison flex nasm
```

Machine learning inference requires ONNX Runtime:

```bash
python3 -m pip install onnxruntime        # CPU
python3 -m pip install onnxruntime-gpu    # CPU + CUDA
```

</details>

<details>
<summary>macOS 13+ (Apple Silicon)</summary>

### Required Packages

These commands assume [Homebrew](https://brew.sh) and Xcode Command Line Tools are already installed. Older macOS versions may work if you install Clang 14 or newer.

> [!IMPORTANT]
> CyberEther does not support the Metal backend on Intel-based Macs. If you are building on an Intel Mac, install the optional Vulkan packages below as a workaround.

Core build and shader tools:

```bash
brew install pkg-config glslang spirv-cross
```

Python build modules:

```bash
python3 -m pip install meson ninja numpy
```

### Optional Features

Remote streaming uses bundled GStreamer sources and requires NASM along with Bison 2.4+:

```bash
brew install bison nasm
export PATH="$(brew --prefix bison)/bin:$PATH"
```

Machine learning inference requires ONNX Runtime:

```bash
brew install onnxruntime
```

Vulkan backend (required for Intel-based Macs):

```bash
brew install molten-vk vulkan-tools vulkan-headers
```

</details>

<details>
<summary>Windows</summary>

### Required Packages

Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) with the C++ workload, [Python 3](https://www.python.org), and the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

If you use Chocolatey, install `pkg-config`:

```bash
choco install pkgconfiglite -y
```

Install the Python build modules:

```bash
python -m pip install meson ninja numpy
```

### Optional Features

Remote streaming uses bundled GStreamer sources and requires NASM and WinFlexBison:

```bash
choco install nasm winflexbison3 -y
```

</details>

<details>
<summary>Browser (Chrome / WebAssembly)</summary>

### Required Packages

All browser runtime dependencies ship with the repository. Make sure [Emscripten](https://emscripten.org/docs/getting_started/downloads.html), [Rust Cargo](https://www.rust-lang.org/tools/install), Python 3, `pkg-config`, `glslangValidator`, and `spirv-cross` are available on your `PATH`.

On Debian or Ubuntu, install the same package set used by continuous integration:

```bash
apt install git build-essential pkg-config python3-pip glslang-tools spirv-cross cargo
python3 -m pip install meson ninja numpy
```

</details>

<details>
<summary>iOS (iPhone & iPad, Experimental)</summary>

### Required Packages

Build iOS targets from macOS with the latest [Xcode](https://developer.apple.com/xcode/) installed, along with the same Homebrew shader tools and Python packages used for macOS:

```bash
brew install pkg-config glslang spirv-cross
python3 -m pip install meson ninja numpy
```

</details>

> [!NOTE]
> Enabling TensorRT inference on Linux requires an ONNX Runtime build with both the CUDA and TensorRT execution providers enabled, plus matching NVIDIA TensorRT libraries in your environment.

## Clone the Repository

Clone CyberEther with all Git submodules initialized:

```bash
git clone --recursive https://github.com/luigifcruz/CyberEther.git
cd CyberEther
```

## Meson Build Options

Pass any of the options from `meson_options.txt` to `meson setup` to customize your build:

| Option | Type | Default | What It Controls |
|---|---|---|---|
| `-Dnative` | boolean | `false` | Enables CPU architecture-specific optimizations (`-march=native`). |
| `-Dpython` | boolean | `false` | Builds the [Superluminal](/docs/superluminal) Python bindings. |
| `-Dwheel` | boolean | `false` | Enables Python wheel packaging mode for Superluminal. |
| `-Dtests` | boolean | `true` | Builds the Catch2 and helper test targets. |
| `-Dexamples` | boolean | `true` | Builds the C++ example programs under `examples/`. |
| `-Dremote` | feature | `auto` | Enables [Remote Streaming](/docs/remote) with bundled static GStreamer. |
| `-Dinference` | feature | `auto` | Enables ONNX Runtime machine learning inference blocks in the ML domain. |
| `-Dterminal` | feature | `auto` | Enables the integrated terminal tile backed by `libghostty-vt`. |

## Configure, Build, and Install

<details>
<summary>Linux or macOS</summary>

Configure a debug-optimized build, compile, and install the binary and libraries:

```bash
meson setup -Dbuildtype=debugoptimized build
meson compile -C build
meson install -C build --skip-subprojects
```

Verify the installation by running:

```bash
cyberether --help
```

</details>

<details>
<summary>Windows</summary>

Run these commands from a Visual Studio Developer PowerShell or any shell with the Visual Studio toolchain loaded:

```bash
meson setup --vsenv -Dbuildtype=debugoptimized build
meson compile -C build
meson install -C build --skip-subprojects
```

Verify the installation by running:

```bash
cyberether --help
```

</details>

<details>
<summary>Browser (Chrome / WebAssembly)</summary>

Cross-compile CyberEther to WebAssembly using the included Emscripten cross file:

```bash
meson setup --cross-file meson/crosscompile/emscripten.ini \
            -Dbuildtype=debugoptimized \
            build-wasm
meson compile -C build-wasm
```

The browser target has no install step. Distributable files are written directly to `build-wasm/`:

```text
build-wasm/cyberether.js
build-wasm/cyberether.wasm
build-wasm/cyberether.js.symbols
build-wasm/cyberether.wasm.map
```

Serve these files from a WebGPU-capable website with cross-origin isolation headers enabled, which Emscripten pthreads require. You can also load your custom `cyberether.js` and `cyberether.wasm` files straight into [cyberether.org/web](https://cyberether.org/web) to test them quickly. A fully offline browser bundle is on the roadmap.

</details>

<details>
<summary>iOS (iPhone & iPad, Experimental)</summary>

> [!NOTE]
> The iOS application target is experimental and not actively maintained for everyday releases.

Cross-compile the library and install it into the Xcode project directory:

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

Once the build completes, open the Xcode project inside `apps/ios/` and deploy it to your device.

</details>

## Running Tests

Make sure `catch2` is available (via your system package manager or Meson subproject) and `-Dtests=true` is set, then run the default test suite:

```bash
meson test -C build
```

The test setup includes four groups:

- **Core, agent, and module Catch2 suites:** Unit and integration tests for tensors, blocks, modules, docs parsing, and compositor state.
- **Shader parser test (`shaders` suite):** A Python test verifying shader preprocessing in `tests/shaders/parser_test.py`.
- **Browser test runner:** In WebAssembly builds, tests run under Node.js (targets are not built by default until requested).
- **Opt-in SDR hardware suite (`sdr` suite):** Excluded from the default test run because it requires physical radio hardware. Run it explicitly with `meson test -C build --suite sdr`.

## Generated Build Macros

Meson generates `jetstream/config.hh` during configuration with preprocessor macros describing the build setup and target operating system.

### Build Configuration Macros

| Macro | Meaning |
|---|---|
| `JETSTREAM_VERSION_STR` | Current library version string. |
| `JETSTREAM_BUILD_TYPE` | Meson build type (`debug`, `debugoptimized`, `release`, `plain`, `minsize`, or `custom`). |
| `JETSTREAM_BUILD_OPTIMIZATION` | Compiler optimization level (`plain`, `0`, `g`, `1`, `2`, `3`, or `s`). |
| `JETSTREAM_BUILD_DEBUG` | Always defined as `"true"` or `"false"`, depending on Meson `debug`. |
| `JETSTREAM_BUILD_NATIVE` | Always defined as `"true"` or `"false"`, depending on `-Dnative`. |
| `JST_IS_STATIC` | Defined for static library builds. |
| `JST_IS_SHARED` | Defined for shared library builds. |
| `JST_DEBUG_MODE` | Defined in builds with `debug=true` (such as `debug` and `debugoptimized`). |
| `JST_RELEASE_MODE` | Defined in release builds (`debug=false`). |

### Target Platform Macros

Exactly one `JST_OS_*` macro is defined for the target platform:

| Macro | Platform |
|---|---|
| `JST_OS_WINDOWS` | Windows (64-bit) |
| `JST_OS_MAC` | macOS |
| `JST_OS_LINUX` | Linux |
| `JST_OS_ANDROID` | Android |
| `JST_OS_IOS` | iOS (iPhone, iPad, tvOS) |
| `JST_OS_BROWSER` | Browser (Emscripten / WebAssembly) |
