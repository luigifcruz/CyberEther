---
title: Quick Start
description: Launch CyberEther and build your first live spectrum analyzer in five minutes.
order: 120
category: Getting Started
---

This guide walks you from a fresh installation to a live spectrum analyzer in just a few minutes, with no radio hardware required.

## Launching CyberEther

After [installing CyberEther](/docs/installation), start it from your terminal:

```bash
cyberether
```

The application opens on the **Start** tab. From there, you can create a blank flowgraph, reopen a recent file, or open a hardware-free gallery example such as **Signal Generator** or **CyberEther First Steps** to see a live signal immediately.

To open a saved flowgraph file directly on startup, pass its path as a positional argument:

```bash
cyberether path/to/flowgraph.yml
```

## Five Core Concepts

Before dropping blocks onto the canvas, keep these five ideas in mind:

- **Blocks** are the nodes on your canvas. Each block performs one focused job, such as generating a waveform, running an FFT, or drawing a plot.
- **Ports** are the small connection points along a block's edges. Output ports send data into input ports.
- **Wires** connect ports and carry data downstream. A block waits to run until every required input is wired up.
- **Fixed shapes** keep the pipeline fast and predictable. Every block processes tensors of a fixed shape determined by its upstream source until another block explicitly reshapes them. The [FAQ](/docs/faq#why-are-buffer-sizes-stricter-than-in-other-sdr-software) explains why CyberEther uses fixed shapes.
- **Inline properties** live directly inside each block node. Click any field to edit it while the flowgraph is running.

## Build Your First Flowgraph

CyberEther runs your flowgraph live as you build it. The moment a block has valid inputs and settings, it starts processing data. You can pause execution anytime using the play and pause button (the first button in the editor toolbar), which freezes every view on its latest frame.

In this walkthrough, you will build a five-block spectrum analyzer by connecting **Signal Generator**, **FFT**, **Amplitude**, **Range**, and **Lineplot**.

### Step 1: Add a Source Block

Open the block browser in whichever way feels more natural:

- Click the **Add block** button (the box with a plus) in the bottom-right toolbar. The new block appears at the center of your view.
- Double-click any empty spot on the canvas. The new block appears right where you clicked.

In both cases, the block browser pops up in the center of the editor. Search for **Signal Generator** and add it to the canvas. It produces synthetic waveforms so you can experiment without external hardware.

> [!TIP]
> If you have a supported radio plugged in, you can drop in an **SDR** block instead to analyze live over-the-air signals. See [Hardware And Devices](/docs/hardware) for supported radios and setup notes.

### Step 2: Add Processing and Display Blocks

Open the block browser again and add these four blocks one at a time using the search box:

1. The **FFT** block converts time-domain samples into the frequency domain.
2. The **Amplitude** block turns complex FFT bins into a magnitude trace.
3. The **Range** block rescales the vertical axis so the signal fits cleanly on screen.
4. The **Lineplot** block renders the trace live.

Drag the blocks into a left-to-right chain in that order, leaving a little space between nodes so the wires stay readable.

### Step 3: Wire the Chain Together

Hover over the output port on **Signal Generator** and drag a wire to the input port on **FFT**. Repeat to link the rest of the pipeline:

- Connect **FFT** output to **Amplitude** input
- Connect **Amplitude** output to **Range** input
- Connect **Range** output to **Lineplot** input

Hover over any wire or output label at any time to inspect the tensor shape and data type moving across that link.

### Step 4: Configure the Parameters

Click inside **Signal Generator** and **FFT** to set these values (hover over any field label to read its tooltip):

| Block | Field | Value |
|---|---|---|
| **Signal Generator** | Data type | `CF32` |
| **Signal Generator** | Signal type | `cosine` |
| **Signal Generator** | Sample rate | `1` MHz |
| **Signal Generator** | Frequency | `0.1` MHz |
| **Signal Generator** | Amplitude | `1` |
| **FFT** | Direction | `forward` |

Leave **Amplitude**, **Range**, and **Lineplot** at their defaults for now (or adjust **Range** to roughly `-100` to `0` dB if the peak looks clipped).

To read the full reference for any block, right-click it and select **Documentation**. The reference opens in a single tile beside the editor.

### Step 5: Watch It Live

Because the pipeline runs continuously, **Lineplot** immediately draws a live spectrum peak for your generated tone at `0.1` MHz. Every parameter tweak or new block updates in real time.

If the plot stays blank or flat, check three things:

- **Confirm the graph is running:** Make sure the play and pause button in the toolbar is active (or press **Space**).
- **Check block states:** Verify that no block shows an **Incomplete** or **Errored** banner, and that **Signal Generator** has **Data type** set to `CF32`.
- **Adjust the vertical range:** Tune the **Range** block (for example `-100` to `0` dB) if the peak sits above or below the visible window.

## Next Steps

- Learn canvas shortcuts, dock tiles, and the built-in example gallery in [The Editor](/docs/editor).
- Understand `.yml` flowgraph files and how data moves along wires in [Flowgraphs](/docs/flowgraphs).
- Explore every built-in node in the [Blocks Catalog](/docs/blocks).
- Learn how to script custom logic in the [Python Block guide](/docs/python-block).
- Stream a headless workstation to your laptop or phone with [Remote Streaming](/docs/remote).
