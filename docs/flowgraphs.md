---
title: Flowgraphs
description: Understand YAML flowgraph files, fixed-shape tensors, wire data, block states, and cross-platform portability.
order: 210
category: Usage
---

A flowgraph is the complete description of a CyberEther pipeline: every block, every parameter, every wire between ports, and the layout of your canvas and detached tiles. Flowgraphs save as human-readable `.yml` files that you can version-control in Git, share with teammates, and open identically across Linux, macOS, Windows, and the web.

## Inside a Flowgraph File

Most users create and edit flowgraphs visually in [The Editor](/docs/editor), where CyberEther automatically manages defaults, device selection, and wire links. Whenever you save, CyberEther writes the version 2 YAML flowgraph schema. Each file starts with optional document metadata (`revision`, `title`, `summary`, `author`, `license`, `category`, `accent`, `diagram`, `compatibility`, `description`) followed by a `graph` list describing every block in the pipeline:

```yaml
---
version: 2
title: Spectrum Analyzer
summary: Receive a band and display a live trace and waterfall.
graph:
  - name: sdr
    module: sdr
    device: cpu
    runtime: native
    provider: generic
    config:
      frequency: 9.49e+07
      sampleRate: 2000000
      numberOfBatches: 8
      numberOfTimeSamples: 2048
    meta:
      node:
        x: 49
        y: 94
        width: 332

  - name: spectrum_analyzer
    module: spectrum_analyzer
    device: cpu
    runtime: native
    provider: generic
    config:
      rangeMin: -100
      rangeMax: 0
    input:
      buffer: '${graph.sdr.output.signal}'
    meta:
      node:
        x: 497
        y: 95
        width: 860
      surface_default:
        attachedHeight: 626
        detached: false
```

Each entry in `graph` captures five parts of a block's state:

- **Identity (`name` and `module`):** A unique instance name within the file and the stable block type identifier listed in the [Blocks Catalog](/docs/blocks).
- **Execution target (`device`, `runtime`, `provider`):** Which compute device and backend implementation the block uses (for example `cpu`, `cuda`, `metal`, or `vulkan`).
- **Parameters (`config`):** The block's inline parameter values.
- **Connections (`input`):** Wires written as `${graph.<producer>.output.<port>}` links pointing from an upstream output port into a downstream input port.
- **Layout (`meta`):** Canvas coordinates, node width, collapsed state, frozen state, and detached surface tile layout.

## What Wires Carry

Every wire connects one output port to one or more input ports and carries a **tensor** by reference rather than copying samples. Alongside the raw sample buffer, each tensor carries:

- **Data type and shape:** For example `CF32` complex floats with shape `[8, 2048]`.
- **Signal axes:** Labels telling downstream blocks which dimension represents batches, channels, or time samples.
- **Tensor attributes:** Metadata such as `sampleRate`, `frequency`, and `sampleScale` that automatically travel downstream so plots can label frequency axes and demodulators know the incoming rate.

Hover over any wire or output port on the canvas to inspect its live shape and data type. For the full metadata model, see [Metadata](/docs/metadata).

## The Fixed-Shape Model

CyberEther allocates tensors at fixed shapes when the graph is created and reuses those exact buffers on every compute cycle. A source block (such as **SDR** or **Signal Generator**) determines the initial buffer shape, and downstream blocks keep that shape until a block explicitly changes it (such as **Reshape**, **Decimator**, **Slice**, or **Pad**).

Because buffer sizes are fixed up front, CyberEther avoids dynamic allocations on the hot path, shares tensor handles along wires, and maps buffers across CPU, GPU, Python, and rendering surfaces with zero copies wherever the underlying devices support it (see [Tensors](/docs/tensors) for the device mapping matrix). If a block reports a shape mismatch, insert a shaping block upstream or adjust the source's batch and sample counts. The [FAQ](/docs/faq#why-are-buffer-sizes-stricter-than-in-other-sdr-software) explains this design trade-off in more detail.

## Block States

While you build and tune a pipeline, each block sits in one of three live states:

| State | Meaning | What to Do |
|---|---|---|
| **Created** | All required inputs are wired, parameters are valid, and the block is actively processing. | Normal operation. |
| **Incomplete** | One or more required input wires are missing, or a hardware device has not been selected yet. | Connect the missing input port or pick a device in the block's fields. |
| **Errored** | Validation failed (for example an unsupported data type, invalid shape, or missing file) or runtime execution returned an error. | Read the status banner on the block, fix the upstream chain or parameter, or use **Reload Block** from the block menu. |

You can also freeze an individual block from its context menu to hold its current output or display in place. Frozen state is preserved when you save the flowgraph.

## Saving, Sharing, and Portability

Flowgraphs are designed to move cleanly between machines and platforms:

- **Cross-platform compatibility:** A `.yml` file saved on Linux opens unchanged on macOS, Windows, and the [Web Edition](/docs/web).
- **Relative and embedded assets:** Keep file paths relative or pair your `.yml` with sample recordings when sharing with others. Inline **Python** block code and **Note** block Markdown are stored directly inside the `.yml` file.
- **Legacy layout migration:** When you open a flowgraph saved by an older version of CyberEther, the old stack layout is discarded and detached panes are positioned from their per-block metadata. Saving the file writes the modern dock tile layout.
- **Opening from the command line:** Pass any `.yml` path directly to `cyberether path/to/flowgraph.yml` on desktop or headless servers. See [Native](/docs/native) for all startup options.
