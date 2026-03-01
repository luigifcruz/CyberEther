---
title: Installation
description: How to install CyberEther on your system.
order: 2
category: Getting Started
---

CyberEther is currently installed by building from source. Follow the instructions below to compile it on your system. More installation methods are coming soon.

<!-- TODO: Add Docker installation section -->

## Build From Source

CyberEther requires a C++20 compiler (GCC 11+ or Clang 14+), [Meson](https://mesonbuild.com), and [Ninja](https://ninja-build.org).

### Dependencies
<!-- [NEW DEPENDENCY HOOK] -->

<details>
<summary>Linux (Arch Linux)</summary>

#### Linux (Arch Linux)
Core dependencies.
```bash
pacman -S git base-devel cmake pkg-config ninja meson git zenity
```

Graphical dependencies for **X11**-based display servers.
```bash
pacman -S glslang glfw-x11
yay -S spirv-cross
```

Graphical dependencies for **Wayland**-based display servers.
```bash
pacman -S glslang glfw-wayland
yay -S spirv-cross
```

Vulkan backend dependencies.
```bash
pacman -S vulkan-icd-loader vulkan-validation-layers
```

#### Optional

SoapySDR block with RTL-SDR support.
```bash
pacman -S soapysdr soapyrtlsdr
```

Remote capabilities.
```bash
pacman -S gstreamer gst-plugins-base gst-libav
pacman -S gst-plugins-good gst-plugins-bad gst-plugins-ugly
```

Examples metadata.
```bash
pacman -S python-yaml
```

</details>

<details>
<summary>Linux (Ubuntu 22.04)</summary>

#### Linux (Ubuntu 22.04)
Core dependencies.
```bash
apt install git build-essential cmake pkg-config ninja-build meson git zenity
```

Graphical dependencies.
```bash
apt install spirv-cross glslang-tools libglfw3-dev
```

Vulkan backend dependencies.
```bash
apt install mesa-vulkan-drivers libvulkan-dev vulkan-validationlayers
```

#### Optional

SoapySDR block with RTL-SDR support.
```bash
apt install libsoapysdr-dev soapysdr-module-rtlsdr
```

Remote capabilities.
```bash
apt install libgstreamer1.0-dev gstreamer1.0-libav \
            gstreamer1.0-plugins-base libgstreamer-plugins-bad1.0-dev \
            libgstreamer-plugins-base1.0-dev libgstreamer-plugins-good1.0-dev \
            gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly
```

Examples metadata.
```bash
apt install python3-yaml
```

</details>

<details>
<summary>macOS 13+ (Apple Silicon)</summary>

#### macOS 13+ (Apple Silicon)
This assumes [Homebrew](https://brew.sh) is installed. Older versions of macOS might work but installing a newer Clang compiler (14+) will be necessary. Metal on Intel-based Macs is not supported by CyberEther. As a workaround, make sure to install the optional Vulkan dependencies listed below.

Core dependencies.
```bash
brew install cmake pkg-config ninja meson
```

Graphical dependencies.
```bash
brew install spirv-cross glslang glfw
```

#### Optional

SoapySDR block with RTL-SDR support.
```bash
brew install soapysdr soapyrtlsdr
```

Remote capabilities.
```bash
brew install gstreamer
```

Examples metadata.
```bash
python -m pip install PyYAML
```

Vulkan backend (required for Intel-based Macs).
```bash
brew install molten-vk vulkan-tools vulkan-headers
```

</details>

<details>
<summary>Browser (Chrome)</summary>

#### Browser (Chrome)

All CyberEther runtime dependencies for the browser are included in the repository. You only need to make sure you have [Python 3](https://www.python.org), [Emscripten](https://emscripten.org/docs/getting_started/downloads.html), and [Rust Cargo](https://www.rust-lang.org/tools/install) installed.

</details>

<details>
<summary>iOS (iPhone & iPad)</summary>

#### iOS (iPhone & iPad)

All CyberEther dependencies for iOS are included in the repository. You only need to make sure you have the latest [Xcode](https://developer.apple.com/xcode/) installed.

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

Build and install.
```bash
meson setup -Dbuildtype=debugoptimized build && cd build
ninja install
```

Done! The executable will be installed in the default terminal path. For usage instructions, run `cyberether --help`.

</details>

<details>
<summary>Browser (Chrome)</summary>

#### Browser (Chrome)

Build project with cross-compilation to WebAssembly.
```bash
meson setup --cross-file meson/crosscompile/emscripten.ini \
            -Dbuildtype=debugoptimized build-browser && \
            cd build-browser
ninja
```

Copy dependencies to the build directory.
```bash
cp ../resources/web/cyberether.html .
cp ../resources/images/cyberether.png .
```

Start the web server and navigate to [http://localhost:8000/cyberether.html](http://localhost:8000/cyberether.html).
```bash
python ../resources/web/local_server.py
```

</details>

<details>
<summary>iOS (iPhone & iPad)</summary>

#### iOS (iPhone & iPad)

Build the project with cross-compilation to iOS and install binaries in the Xcode project.
```bash
meson setup --cross-file meson/crosscompile/ios.ini \
            --prefix $(pwd)/apps/ios/CyberEtherMobile/Library \
            -Dbuildtype=debugoptimized build-ios && \
            cd build-ios
ninja install
```

After the build is complete, open the Xcode project in `apps/ios/` and run it on your device.

</details>
