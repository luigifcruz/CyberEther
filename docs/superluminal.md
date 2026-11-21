---
title: Superluminal
description: Embed CyberEther GPU plots, layouts, interactive widgets, and remote displays from C++ and Python.
order: 350
category: Development
---

Superluminal is CyberEther's high-level plotting library for standalone C++ programs and Python scripts. Reach for it whenever you already have a simulation, hardware test bench, or data analysis script and want live GPU-accelerated signal plots without opening the full flowgraph editor.

## Plot Types and Layouts

Superluminal arranges plots and control panels on a grid called a **mosaic** (a 2D matrix of `0` and `1` cells marking where a panel sits). You can build grids manually or use `Superluminal::MosaicLayout(matrixHeight, matrixWidth, panelHeight, panelWidth, offsetX, offsetY)` in C++ (or `sl.layout(...)` in Python).

| Plot Type | C++ Enum | Python Constant | Best For |
|---|---|---|---|
| **Line** | `Superluminal::Type::Line` | `sl.line` | 1D waveforms, spectra, and magnitude traces. |
| **Heat** | `Superluminal::Type::Heat` | `sl.heat` | 2D matrices, spectrograms, and heat maps. |
| **Scatter** | `Superluminal::Type::Scatter` | `sl.scatter` | Complex I/Q constellations and scatter clouds. |
| **Waterfall** | `Superluminal::Type::Waterfall` | `sl.waterfall` | Scrolling spectrum history over time. |
| **Spectrum Analyzer** | `Superluminal::Type::SpectrumAnalyzer` | `sl.spectrum_analyzer` | Combined live spectrum trace and scrolling waterfall. |

Each plot also configures its source and display domains (`Time` or `Frequency`) and complex-to-real operation (`Real`, `Imaginary`, `Amplitude`, or `Phase`), plus optional `batchAxis`, `channelAxis`, and `channelIndex` selectors.

## Using Superluminal From C++

Include `<jetstream/superluminal.hh>`, initialize the window, attach plots to your tensors, and display either a static snapshot (`Show()`) or a live updating loop (`RealtimeLoop()`):

```cpp
#include <cmath>
#include <jetstream/superluminal.hh>

using namespace Jetstream;

int main() {
    Superluminal::InstanceConfig config;
    config.windowTitle = "Live Spectrum";
    config.interfaceSize = {1280, 720};
    Superluminal::Initialize(config);

    Tensor buffer;
    buffer.create(DeviceType::CPU, DataType::CF32, {2048});

    Superluminal::Plot("Spectrum", {{1}}, {
        .buffer = buffer,
        .type = Superluminal::Type::Line,
        .source = Superluminal::Domain::Time,
        .display = Superluminal::Domain::Frequency,
        .operation = Superluminal::Operation::Amplitude,
    });

    Superluminal::RealtimeLoop([&](bool& running) {
        F32 phase = 0.0f;
        while (running && Superluminal::Presenting()) {
            auto* data = buffer.data<CF32>();
            for (U64 i = 0; i < 2048; ++i) {
                const F32 angle = 2.0f * 3.14159265f * 0.05f * (static_cast<F32>(i) + phase);
                data[i] = {std::cos(angle), std::sin(angle)};
            }
            phase += 2048.0f;
            Superluminal::Update("Spectrum");
        }
    });

    return 0;
}
```

### Interactive Widgets

Add live controls right beside your plots using `Box`, `Interface`, or `GlobalInterface` callbacks:

- **Text and Markdown:** Call `Superluminal::Text("Peak: {:.2f} dB", peak)` or `Superluminal::Markdown(content)` to display live readouts and notes.
- **Sliders:** Call `Superluminal::Slider("Gain", 0.0f, 10.0f, gain)` (accepting either `F32&` or `U64&`) to tune variables live while your loop runs.

## Using Superluminal From Python

Build the Python bindings with `-Dpython=true` (add `-Dwheel=true` as well when packaging a Python wheel, as covered in [Building From Source](/docs/building)). Then import `superluminal` and pass NumPy arrays directly to `plot`:

```python
import numpy as np
import superluminal as sl

buf = np.zeros(2048, dtype=np.complex64)
sl.plot(buf, sl.line, label="Spectrum", domain=(sl.time, sl.frequency))

def worker():
    phase = 0.0
    while sl.running():
        t = np.arange(2048, dtype=np.float32) + phase
        buf[...] = np.exp(1j * 2.0 * np.pi * 0.05 * t).astype(np.complex64)
        phase += 2048.0
        sl.update("Spectrum")

sl.realtime(worker)
```

Python exposes the same layout and widget helpers (`sl.configure`, `sl.layout`, `sl.box`, `sl.text`, `sl.slider`, `sl.markdown`) as C++.

> [!IMPORTANT]
> The Python Superluminal binding currently accepts **C-contiguous CPU NumPy arrays** of type `float32` (`F32`) or `complex64` (`CF32`) only. Passing a CUDA array (such as a CuPy `ndarray`) raises an error in the current binding. Convert GPU arrays to CPU NumPy before passing them to `sl.plot`.

## Remote Display in Superluminal

Superluminal supports the same self-hosted WebRTC [Remote Streaming](/docs/remote) stack as the main application, letting headless C++ or Python scripts stream their live plots to any browser.

Enable it on `InstanceConfig` (or via `sl.configure(...)` in Python):

- Set `remote = true` along with optional `remoteCodec` (`"h264"`, `"vp8"`, `"vp9"`), `remoteEncoder` (`"auto"`), `remoteAutoApprove`, and `remoteFramerate` (default `30`).
- Retrieve connection details programmatically with `Superluminal::RemoteRoomId()`, `RemoteInviteUrl()`, `RemoteAccessToken()`, or print the invite and QR code to your terminal with `Superluminal::PrintRemoteInfo()` (in Python: `sl.remote_room_id()`, `sl.remote_invite_url()`, `sl.remote_access_token()`, and `sl.print_remote_info()`).

See `examples/superluminal/` in the repository for complete working examples covering `hello_world`, `realtime`, `constellation`, `interface`, `global_interface`, `interferometer`, and `remote`.
