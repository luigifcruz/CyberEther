---
title: Quick Start
description: Launch CyberEther, run your first flowgraph, and explore the built-in examples.
order: 3
category: Getting Started
---

This guide gets you from installation to a running flowgraph in a few minutes.

## Launching CyberEther

After [installing CyberEther](/docs/installation), launch it from a terminal:

```bash
cyberether
```

This opens the main window on the Start tab. From there you can create a new flowgraph, open a saved or recent one, or pick an example from the gallery.

To load a flowgraph directly at startup, pass its path as the single positional argument:

```bash
cyberether examples/flowgraphs/spectrum-analyzer.yml
```

## Key ideas

Before building your first flowgraph, it helps to know a few terms that appear throughout CyberEther.

- **Blocks** are the nodes on the canvas. Each one performs one operation, such as generating a signal, running an FFT, or drawing a plot.
- **Ports** are the small connection points on the edges of a block. Output ports feed data into input ports.
- **Wires** carry data from one block to the next. A block cannot run until its required inputs are connected.
- **Shapes** are fixed. Every block runs on tensors of a fixed shape that flows downstream from the source, so the sample count you set on a source block sets the size of everything downstream until a block reshapes it. The [FAQ](/docs/faq#why-are-buffer-sizes-stricter-than-in-other-sdr-software) explains why.
- **Properties** let you change how a block behaves. They appear inline inside each block; click a field to edit it.

## Creating a simple flowgraph

The CyberEther editor is a node-based flowgraph editor. You build a pipeline by adding blocks, wiring their ports together, and tuning each block's parameters. The flowgraph runs while you edit it. As soon as a block has everything it needs, it begins processing live data. The play and pause button, the first one in the editor toolbar, freezes processing while the views keep their last data.

### 1. Add a source block

Open the block browser in one of two ways:

- Click the **Add block** button (the box with a plus) in the toolbar at the bottom right of the flowgraph editor. The new block lands at the center of the view.
- Double-click anywhere on the empty canvas. The new block lands where you clicked.

Either way, the browser opens in the center of the flowgraph editor.

Find and add a **Signal Generator** block. This block produces a synthetic signal so you do not need any hardware to get started. If you have an RTL-SDR, Airspy, Airspy HF+, HackRF, LimeSDR, bladeRF, or PlutoSDR, use an **SDR** block instead to work with a real radio signal.

### 2. Add processing blocks

Open the block browser again and add one block at a time. Type the block name in the search box to find it quickly, then add it to the canvas. Add these four blocks:

- **FFT**: converts the signal to the frequency domain.
- **Amplitude**: converts the complex FFT output to a magnitude trace.
- **Range**: adjusts the vertical scale of the trace.
- **Line Plot**: draws the trace on screen.

After each block appears on the canvas, drag it into a left-to-right chain: **Signal Generator**, then **FFT**, **Amplitude**, **Range**, and finally **Line Plot**. Leave some space between blocks so the wires are easy to see.

### 3. Connect the blocks

Hover over the output port of the **Signal Generator** block, then drag a wire to the input port of the **FFT** block. Do the same between:

- **FFT** output to **Amplitude** input
- **Amplitude** output to **Range** input
- **Range** output to **Line Plot** input

The wires show how data flows through the pipeline. Hover over a wire or an output label to see the tensor shape and data type carried by that link.

### 4. Configure the blocks

Each block shows its configurable fields directly inside the node on the canvas. Click a field to change its value. Hover over a field label to see a tooltip describing what it does. For example:

- **Signal Generator**
  - Signal type: `cosine`
  - Sample rate: `1e6`
  - Frequency: `1e5`
  - Amplitude: `1`
- **FFT**
  - Direction: `forward`

You can leave the other blocks at their defaults for now. For the complete reference on any block, right-click the block and select **Documentation**.

Documentation opens in one tile beside the editor. Opening another block's documentation replaces that tile's contents; the tile is not saved with the flowgraph layout.

### 5. Watch it run

Because the flowgraph is always running, the **Line Plot** block starts showing a live spectrum of the generated tone as soon as the blocks are wired and configured. If the trace looks too small or clipped, adjust the **Range** block until the signal is clearly visible. If you change a property or add another block, the pipeline updates in real time.

### Navigating the editor and arranging tiles

Click a block to select it. Shift-click adds blocks to the selection, and Shift-drag on empty canvas selects a group. Clicking or panning empty canvas clears the selection. Use **Cmd/Ctrl + Plus**, **Minus**, or **0** to zoom in, zoom out, or reset the editor zoom. Zoom lasts for the current session and is not saved.

Plots inside a block take the same mouse controls as detached plots, so you can zoom, pan, and place markers in place. Dragging over a plot interacts with it, so move the block by its header instead. Click the detach icon in the block header, or choose Detach Plot from the block menu, to move its plots into their own tiles.

Detached plots and configuration panes use dock tiles. Layouts saved by older versions discard their old stack arrangement when opened; detached panes are placed using their per-block metadata. Saving writes the new tile layout, not the old stacks.

The latency overlay setting and graphics framerate control are no longer available. The interface redraws only when something changes.

## Ready-to-go example flowgraphs

Open the examples from the gallery on the Start tab. The gallery downloads the published flowgraphs the first time it is online and keeps a copy for offline use. Each opens in a fresh tab and starts processing once its inputs and devices are ready.

| Flowgraph | What it demonstrates |
|-----------|----------------------|
| CyberEther First Steps | Finish a spectrum analyzer from a lone Signal Generator. Add a display, wire it, edit live settings, read block documentation, and save. |
| Signal Generator | Adds noise to a generated tone and displays the result in a spectrum plot and waterfall. |
| Simple FM Receiver | Receives an FM broadcast station with an RTL-SDR, demodulates the signal, and plays the audio. |
| Multi-FM Receiver | Uses the `Filter` block to separate two FM stations from one RTL-SDR input. Displays both spectra and plays audio from one station. |
| Python Block | Generates noisy QPSK signals in a Python block and displays their spectrum, waterfall, and constellation. Requires [Python with NumPy](/docs/python-block#choosing-a-python-runtime). |
| Overlap-Add-Fold | Explains FFT filtering and resampling in an RTL-SDR FM receiver, with notes on zero padding, spectral folding, and overlap-add reconstruction. |
| Spectrum Analyzer | Applies a window and FFT to RTL-SDR samples and displays the spectrum in a line plot and waterfall. |
| ADS-B Flight Tracker | Receives ADS-B signals at 1090 MHz from an SDR and decodes aircraft positions and flight data. |

Start with **CyberEther First Steps** to turn a lone Signal Generator into a spectrum analyzer in about ten minutes. It needs no radio hardware or Python installation.
