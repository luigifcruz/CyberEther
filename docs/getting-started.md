---
title: Quick Start
description: Launch CyberEther and run your first flowgraph.
order: 3
category: Quick Start
---

This guide walks through launching CyberEther for the first time and running a built-in example flowgraph. Before continuing, make sure CyberEther is [installed](/docs/installation). 

## Launching CyberEther

After installation, start CyberEther from a terminal:

```bash
cyberether
```

![CyberEther welcome screen](/docs/assets/images/welcome-screen.png)

You'll be greeted by the welcome screen with three quick-start actions and a keyboard shortcut reference:

- **New Flowgraph** — start with a blank canvas
- **Open File** — load a flowgraph file from disk
- **Examples** — browse and load the bundled example flowgraphs
- **Keyboard shortcuts** — `Ctrl+N` New, `Ctrl+O` Open, `Ctrl+S` Save, `Ctrl+T` Spotlight

### Status HUD

In the bottom-left corner you'll see a small overlay with three pieces of information:

- **Hz** — the UI render rate. Shown in green when above 50 Hz, meaning the renderer is keeping up. A lower number can indicate the system is under heavy load.
- **Windowing backend** — the window and input system in use (e.g. `GLFW`).
- **Graphics API** — the graphics backend CyberEther selected for your hardware (e.g. `Metal` on Apple Silicon, `Vulkan` on Linux/Windows).
- **Device** — the GPU and memory detected on your system (e.g. `Apple M4 Pro (24 GB)`).

For example, `120 Hz  GLFW (Metal)` on the first line and `Apple M4 Pro (24 GB)` on the second means CyberEther is rendering at 120 frames per second using Metal on an Apple M4 Pro. This is a good sign that everything is working correctly.

## Loading the Signal Generator Example

The Signal Generator example is a good first flowgraph to run as a sanity check as it requires no SDR hardware. 

**Option A — via the Examples button:**

1. Click **Examples** on the welcome screen (or go to **Flowgraph → Open Examples** in the menu bar)
2. Find **Signal Generator** in the list and click it

**Option B — via Open:**

1. Click **Open** (or press `Ctrl+O`)
2. Navigate to your cloned CyberEther repository
3. Open `examples/flowgraphs/signal-generator.yml`

The flowgraph will load and begin running immediately.

## Starting and Stopping Flowgraphs

Flowgraphs run continuously from the moment they are loaded — there is no global pause button. To stop processing:

- **Close the flowgraph:** `Ctrl+W` or **Flowgraph → Close**. 

### Reloading Individual Blocks

If a single block gets into a bad state (shown by a skull icon ☠ or a diagnostic warning), you can restart just that block without closing the entire flowgraph:

1. **Right-click** the block node in the flowgraph editor
2. Select **Reload Block**

This tears down and recreates only that block, leaving the rest of the pipeline running.

### Adding Blocks to FlowGraph

To add a new block to an open flowgraph, **double-click on empty canvas space** to open the block picker. Search for a block by name, select it, and press **Enter** or double-click to place it. Connect blocks by dragging from an output port to an input port.

## Building Your First Flowgraph — FIR Filter

This section walks through building a FIR filter pipeline from scratch. No SDR hardware is required.

### What We're Building

![FIR filter flowgraph](/docs/assets/images/fir-filter-flowgraph.png)

Two signal generators each produce a cosine tone at different frequencies. We combine them, then use a FIR bandpass filter to isolate only one of the two tones. The lineplot will show both peaks before the filter and only one after.

### Step 1 — Open a New Flowgraph

Press `Ctrl+N` or click **New Flowgraph** from the welcome screen.

### Step 2 — Add the First Signal Generator 

**Double-click** on the empty canvas to open the block picker. Type `signal generator` and press **Enter** to place a **Signal Generator** block. Set its config:

| Parameter | Value |
|---|---|
| Signal Type | `cosine` |
| Frequency | `.1` |
| Sample Rate | `1` (Normalize Fs=1) |
| Buffer Size | `8192` |
| Output Data Type | `CF32` |

### Step 3 — Add the Second Signal Generator

**Double-click** again and add a second **Signal Generator** with:

| Parameter | Value |
|---|---|
| Signal Type | `cosine` |
| Frequency | `.25` (0.25*Fs) |
| Sample Rate | `1` |
| Buffer Size | `8192` |
| Output Data Type | `CF32` |

### Step 4 — Add an Add Block

**Double-click** and add an **Add** block (Data Type: `CF32`). Then wire:
- Cosine generator **Output** → Add **A** input
- Noise generator **Output** → Add **B** input

### Step 5 — Add and Configure the FIR Filter

**Double-click** and add a **Filter** block. Connect the **Add** block's **Output** → **Filter**'s **Signal** input.

The `filter` block is an all-in-one FIR bandpass filter — it generates and applies its own coefficients. Configure it to isolate the 0.1*Fs tone and reject the 0.25*Fs tone:

| Parameter | Value | Notes |
|---|---|---|
| Sample Rate | `1` | Must match the signal source |
| Bandwidth | `0.20` | 0.2*Fs passband |
| Center | `0.0` | Centered on our cosine tone |
| Taps | `201` | More taps = sharper cutoff, more latency |

### Step 6 — Add a Spectrum Engine

**Double-click** and add a **Spectrum Engine** block. Connect **Filter**'s **Buffer** output → **Spectrum Engine**'s **Input**.

| Parameter | Value |
|---|---|
| Axis | `1` |
| Enable Scale | `true` |
| Range Min | `-200` |
| Range Max | `0` |

### Step 7 — Add a Lineplot

**Double-click** and add a **Lineplot** block. Connect **Spectrum Engine**'s **Output** → **Lineplot**'s **Signal** input.

> **Tip:** If the plot appears blank, the block may be too small to render. Drag the bottom or right edge of the Lineplot node to make it larger, the plot will appear once the block has enough space.

You should now see only the 0.1*Fs tone — the 0.25*Fs tone is attenuated by the filter. Press `Ctrl+S` to save your flowgraph. **Make sure you save it with .yml extension**


