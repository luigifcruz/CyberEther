---
title: The Editor
description: Navigate the canvas, manage dock tiles, use keyboard shortcuts, and explore built-in workbench tools and examples.
order: 200
category: Usage
---

The CyberEther editor is a live, GPU-rendered workbench where you assemble flowgraphs, tune parameters, inspect signals, and organize views into dockable tiles. Everything runs continuously as you edit, so changes appear immediately without a separate compile or run step.

## Canvas Controls and Shortcuts

Every flowgraph opens in its own tab. Use these controls to move around the canvas and edit nodes comfortably:

| Action | Control |
|---|---|
| Open the block browser | Double-click empty canvas, or click **Add block** in the bottom-right toolbar |
| Add a recording as a File Reader | Drag and drop a recording file directly onto the canvas |
| Pause or resume flowgraph | **Space** (or the play and pause button in the toolbar) |
| Select a block | Click the block |
| Add blocks to selection | **Shift + Click** additional blocks |
| Select a group | **Shift + Drag** across empty canvas |
| Clear selection | Click or pan empty canvas |
| Cut, copy, paste, duplicate, delete | **Cmd/Ctrl + X**, **C**, **V**, **D**, and **Delete** |
| Freeze or unfreeze selected block | **Cmd/Ctrl + B** (or **Freeze Block** in the block menu) |
| Undo and redo edits | **Cmd/Ctrl + Z** and **Cmd/Ctrl + Shift + Z** (or **Cmd/Ctrl + Y**) |
| Open, save, save as | **Cmd/Ctrl + O**, **Cmd/Ctrl + S**, and **Cmd/Ctrl + Shift + S** |
| New tab or close tab | **Cmd/Ctrl + T** and **Cmd/Ctrl + W** |
| Open Settings | **Cmd/Ctrl + Comma** |
| Zoom canvas in, out, or reset | **Cmd/Ctrl + Plus**, **Minus**, or **0** (session-only, not saved) |
| Move a block with an inline plot | Drag the block header |
| Zoom, pan, or place markers on a plot | Scroll, drag, or **Shift + Click** directly inside the plot |
| Detach a plot into its own tile | Click the detach icon in the block header, or choose **Detach Plot** from the block menu |
| Open in-app documentation | **F1** |
| Search inside documentation viewer | **Cmd/Ctrl + K** (when the documentation tab is active) |
| Toggle integrated terminal | **Ctrl + Backtick** (when built with terminal support) |
| Toggle Assistant panel | **Cmd/Ctrl + L** (when enabled in Settings) |
| Toggle Inspector, timing, reload plugins, or send feedback | **Cmd/Ctrl + I**, **Cmd/Ctrl + Alt + T**, **Cmd/Ctrl + Alt + R**, and **Cmd/Ctrl + G** |

Plots inside a block support the same mouse interactions as detached plots. Because dragging inside a plot pans or zooms the visualization itself, grab the block header whenever you want to reposition the node on the canvas.

## Running, Pausing, and Freezing

CyberEther gives you two levels of execution control while inspecting live signals:

- **Pause the flowgraph:** Press **Space** or click the play and pause button (the first button in the editor toolbar) to pause the entire compute loop and freeze every view on its latest frame.
- **Freeze an individual block:** Right-click a visualization or processing block and choose **Freeze Block** (or press **Cmd/Ctrl + B**) to freeze just that node while the rest of the pipeline continues running.

## Dock Tiles and Workspace Panels

Detached plots, configuration panes, and workbench tools live in dockable tiles that you can split, resize, and rearrange around the canvas:

- **Detached plots:** Move any plot into its own tile for a larger view while keeping the canvas tidy. Detached tile positions are saved with your flowgraph.
- **Block documentation tile:** Right-click any block and choose **Documentation** to open its built-in reference beside the editor. Opening docs for another block replaces that tile, and documentation tiles are not saved with your flowgraph.
- **Environment tile:** Open the **Environment** tile to inspect live key-value pairs shared across the flowgraph by C++ and Python blocks. See [Metadata](/docs/metadata) for how blocks publish environment data.
- **In-app documentation viewer:** Press **F1** to browse this full documentation set offline inside CyberEther, or press **Cmd/Ctrl + K** inside the viewer to search headings and body text across every page.
- **Integrated terminal:** Native builds with terminal support enabled include a built-in terminal tile powered by `libghostty-vt` (**Ctrl + Backtick**), letting you run scripts or inspect devices without leaving the app.
- **Assistant panel:** Opt in under **Settings** > **Intelligence** to open the **Assistant** tile (**Cmd/Ctrl + L**) or enable the built-in MCP server. See [Intelligence](/docs/intelligence).

## Settings

Open **Settings** from the titlebar (or press **Cmd/Ctrl + Comma**) to configure CyberEther across its sidebar sections:

- **General:** Appearance (Theme, Interface Scale, Background Particles, Show Tips), Input (Touchpad Scroll Speed), and Rendering (Graphics Renderer).
- **Remote:** Self-hosted WebRTC streaming settings, codec selection, and access controls. See [Remote Streaming](/docs/remote).
- **Maps:** High-resolution map rendering options for blocks such as **ADS-B Decoder**.
- **Runtime:** Choose which Python interpreter runs **Python** blocks and configure PEP 723 dependency installation policies. See [Python Runtime And Dependencies](/docs/python-runtime).
- **Registry:** Register, inspect, reload, and remove `.cep` plugin bundles. See [Installing Plugins](/docs/installing-plugins).
- **Intelligence:** Optional Assistant providers, models, voice mode, and the built-in MCP server. See [Intelligence](/docs/intelligence).
- **Developer:** Show Timing, Show Render Stats, Log Level, Benchmarks, and Plugins reload controls.
- **About:** Build version and application details.
- **Legal:** License terms and third-party acknowledgments.

## Live Markdown in Note Blocks

Drop a **Note** block onto any canvas to document a pipeline or build a live dashboard. Note blocks support standard Markdown along with live `${env.<path>}` placeholders, `[*]` list wildcards for dynamic tables, and fenced `stats` tiles that update in real time from the flowgraph environment. See [Live Markdown in Note Blocks](/docs/metadata#live-markdown-in-note-blocks) for the complete syntax and examples.

## Built-In Example Gallery

Open ready-made pipelines from the gallery on the **Start** tab. The first time you browse the gallery online, CyberEther downloads the published flowgraphs and keeps a local copy for offline use. Each example opens in its own tab and begins running once its inputs and devices are ready.

| Example | What You Will See |
|---|---|
| **CyberEther First Steps** | Interactive tutorial starting from a single Signal Generator. Add a display, wire it up, tune live settings, open block docs, and save. |
| **Signal Generator** | Combines a synthetic tone with noise and plots the result on a spectrum trace and waterfall. |
| **Simple FM Receiver** | Tunes an FM broadcast station on an RTL-SDR, demodulates it, and plays live audio. |
| **Multi-FM Receiver** | Uses the `Filter` block to extract two FM stations from one RTL-SDR stream, showing both spectra while playing audio from one. |
| **Python Block** | Generates noisy QPSK signals in Python and displays spectrum, waterfall, and constellation views. Requires [Python with NumPy](/docs/python-runtime#choosing-a-python-runtime). |
| **Overlap-Add-Fold** | Walks through FFT filtering and resampling in an RTL-SDR FM receiver, with notes on zero padding, spectral folding, and overlap-add reconstruction. |
| **Spectrum Analyzer** | Applies a window and FFT to live RTL-SDR samples and plots them on a line plot and waterfall. |
| **ADS-B Flight Tracker** | Receives 1090 MHz ADS-B transmissions from an SDR and decodes live aircraft positions and flight data. |

> [!TIP]
> If you want a guided ten-minute tour inside the app itself, open **CyberEther First Steps**. It requires no radio hardware and no Python setup.
