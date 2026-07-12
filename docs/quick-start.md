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

This opens the main window with an empty workspace. From there you can build a flowgraph visually or load an existing one.

To load a flowgraph directly at startup, pass its path as the single positional argument:

```bash
cyberether examples/flowgraphs/signal-generator.yml
```

## Key ideas

Before building your first flowgraph, it helps to know a few terms that appear throughout CyberEther.

- **Blocks** are the nodes on the canvas. Each one performs one operation, such as generating a signal, running an FFT, or drawing a plot.
- **Ports** are the small connection points on the edges of a block. Output ports feed data into input ports.
- **Wires** carry data from one block to the next. A block cannot run until its required inputs are connected.
- **Properties** let you change how a block behaves. They appear inline inside each block; click a field to edit it.

## Creating a simple flowgraph

The CyberEther editor is a node-based flowgraph editor. You build a pipeline by adding blocks, wiring their ports together, and tuning each block's parameters. The flowgraph is always running as you edit it. There is no start or stop button. As soon as a block has everything it needs, it begins processing live data.

### 1. Add a source block

Open the block browser in one of two ways:

- Click the **Blocks** button in the toolbar at the top of the flowgraph window.
- Double-click anywhere on the empty canvas.

Find and add a **Signal Generator** block. This block produces a synthetic signal so you do not need any hardware to get started. If you have an RTL-SDR, use a **SoapySDR** block instead to work with a real radio signal.

### 2. Add processing blocks

Open the block browser again and add one block at a time. Type the block name in the search box to find it quickly, then add it to the canvas. Add these four blocks:

- **FFT**: converts the signal to the frequency domain.
- **Amplitude**: converts the complex FFT output to a magnitude trace.
- **Range**: adjusts the vertical scale of the trace.
- **Line Plot**: draws the trace on screen.

After each block appears on the canvas, drag it into a left-to-right chain: **Signal Generator**, then **FFT**, **Amplitude**, **Range**, and finally **Line Plot**. Leave some space between blocks so the wires are easy to see.

### 3. Connect the blocks

Hover over the output port of the **Signal Generator** block, then drag a wire to the input port of the **FFT** block. Do the same between:

- **FFT** output → **Amplitude** input
- **Amplitude** output → **Range** input
- **Range** output → **Line Plot** input

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

### 5. Watch it run

Because the flowgraph is always running, the **Line Plot** block starts showing a live spectrum of the generated tone as soon as the blocks are wired and configured. If the trace looks too small or clipped, adjust the **Range** block until the signal is clearly visible. If you change a property or add another block, the pipeline updates in real time.

## Ready-to-go example flowgraphs

CyberEther ships with a set of built-in example flowgraphs. Open them from the UI by choosing **Flowgraph** → **Open Examples**. The examples modal shows a grid of ready-to-run flowgraphs. Click one to open it in a fresh tab. These examples are the fastest way to see what CyberEther can do:

| Flowgraph | What it demonstrates |
|-----------|----------------------|
| Signal Generator | Generates synthetic signals and visualizes them with a waterfall and line plot. |
| Spectrum Analyzer | A classic SDR-style spectrum analyzer using a windowed FFT. |
| Simple FM Receiver | Receives an FM broadcast station with an RTL-SDR and plays the audio. |
| Multi-FM | Receives multiple FM stations at once using the `Filter Engine` block. |
| Overlap-Add | Frequency-domain FIR filtering using the overlap-add method. |
| Overlap-Add-Fold | Overlap-add filtering with folding and resampling. |
| ADS-B Flight Tracker | Receives ADS-B signals and tracks nearby aircraft. |
