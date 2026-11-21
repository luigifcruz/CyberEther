---
title: Python Block
description: Run custom Python compute code inside a flowgraph with zero-copy tensor access.
order: 230
category: Usage
---

The Python block runs your own Python code on every compute cycle with zero-copy access to flowgraph tensors, tensor metadata, block metrics, and the shared flowgraph environment. Reach for it whenever you want to prototype an algorithm quickly, glue two blocks together with custom logic, or bring NumPy, SciPy, and CuPy straight into a live pipeline without writing C++.

- Need to pick a Python interpreter or declare PEP 723 packages? See [Python Runtime And Dependencies](/docs/python-runtime).
- Ready for CuPy synchronization, tensor attributes, `ctx.env`, `ctx.metrics`, and background worker threads? See [Python Context API](/docs/python-context).

## The `ctx` Object at a Glance

Your script defines a `compute(ctx)` function. On every cycle, CyberEther passes a `ctx` object exposing six attributes:

| Attribute | Access | What It Holds |
|---|---|---|
| `ctx.inputs[i]` | Read-only | Zero-copy NumPy (CPU) or CuPy (CUDA) view of input port `i`. |
| `ctx.outputs[i]` | Writable in place | Zero-copy NumPy (CPU) or CuPy (CUDA) view of output port `i`. |
| `ctx.input_attrs[i]` | Read-only | Dictionary of attributes attached to input tensor `i`. |
| `ctx.output_attrs[i]` | Writable | Dictionary of attributes published on output tensor `i`. |
| `ctx.env` | Read / Write | Graph-wide key-value store shared across all blocks and the Environment tile. |
| `ctx.metrics` | Read-only | Subscription-based view of metrics published by other blocks. |

## Quick Start

Add a **Python** block to your flowgraph (make sure you have a [Python runtime configured](/docs/python-runtime#choosing-a-python-runtime)), set **Input Count** to `1` and **Output Count** to `1`, and write a `compute(ctx)` function:

```python
def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0] * 2.0
```

Your code compiles immediately before its first compute cycle once the flowgraph is scheduled, and `compute(ctx)` runs once per cycle after that. Editing code inline reloads the script in place. Changing the input or output counts recreates the block.

### Loading From an External File

To edit scripts in your own editor, switch **Source** from inline code to **File** and pick a `.py` file. Switching sources stops the previous script while preserving both your inline code and file path. After saving edits to the file, click **Reload** on the block to apply them.

## Block Configuration

| Field | Purpose |
|---|---|
| **Source** | Choose between the inline code editor and an external `.py` file. |
| **Code** | Inline Python source defining `compute(ctx)`. |
| **File** | Path to an external Python script. |
| **Input Count** | Number of input ports (`input0`, `input1`, and so on). |
| **Output Count** | Number of output ports (`output0`, `output1`, and so on). |
| **Output Tensor Specs** | Shape, data type, device, and signal axes for each output port. |

Every output tensor is allocated up front from its spec. Python code cannot change an output's shape, data type, or device at runtime.

- **Supported data types:** `F32`, `CF32`, `F64`, `CF64`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`.
- **Supported devices:** `cpu` and `cuda`.
- **Port counts:** Zero-input sources and zero-output sinks are both supported.

### Declaring Signal Axes

Downstream blocks validate their inputs during graph construction, before `compute(ctx)` ever runs. Because of that, a rank-two or higher Python output cannot wait until `compute(ctx)` to publish `sampleAxis`.

Declare axis roles directly in the output tensor spec using `[B, C, S]` notation, where each comma-separated token assigns a role to that dimension (`B` for batch, `C` for channel, `S` for sample, and `_` for no role). Trailing axes can be omitted, and no role may repeat:

| Spec | Resulting Attributes |
|---|---|
| `[S]` | Rank-one signal (`sampleAxis=0`). |
| `[C, S]` | Channel-major stream (`channelAxis=0`, `sampleAxis=1`). |
| `[B, C, S]` | Batched channels of samples (`batchAxis=0`, `channelAxis=1`, `sampleAxis=2`). |
| `[B, _, S]` | Batched samples with an unlabeled middle dimension. |

Declared axes are published as `sampleAxis`, `batchAxis`, and `channelAxis` when the block is created, so downstream validation sees them immediately. Blank positions receive no attribute. Your script can still read the declared roles from `ctx.output_attrs[i]`, though they are immutable. Attempts to overwrite `sampleAxis`, `batchAxis`, or `channelAxis` inside `compute(ctx)` are ignored with a console warning.

## Four Tensor Rules

Keep four rules in mind whenever your script touches `ctx.inputs` or `ctx.outputs`:

- **Always write outputs in place.** Use slice assignment (`ctx.outputs[0][...] = value`) to write into the downstream buffer. Rebinding a local variable (`out = value`) writes nothing to the flowgraph.
- **Inputs are read-only.** NumPy enforces read-only buffers on CPU tensors, and CUDA inputs are wrapped in a read-only array wrapper. Copy an input before mutating it.
- **Non-contiguous inputs work out of the box.** Strided views from blocks like `slice` or `permutation` arrive as properly strided arrays with zero copies.
- **Devices can be mixed freely.** A single Python block can consume both CPU and CUDA inputs and emit outputs on either device, regardless of the block's own device setting.

## Controlling Execution With `SKIP` and `YIELD`

### Skipping a Cycle With `SKIP`

Return the reserved global `SKIP` whenever your block does not have a valid output ready for the current cycle:

```python
def compute(ctx):
    if not has_complete_symbol(ctx.inputs[0]):
        return SKIP

    ctx.outputs[0][...] = decode_symbol(ctx.inputs[0])
```

Returning `SKIP` emits an empty frame, causing downstream blocks to drop their frames for that step without running. Your output allocations remain intact and the block executes again on the next frame. Returning `None` (or omitting a return statement) completes the cycle normally.

### Pacing Execution With `YIELD`

Return the reserved global `YIELD(seconds)` when your block is not ready to consume its inputs yet, such as a playback script running ahead of wall-clock time:

```python
import time

INTERVAL = 0.01
due = 0.0

def compute(ctx):
    global due
    now = time.monotonic()
    if now < due:
        return YIELD(due - now)

    due = max(due, now) + INTERVAL
    ctx.outputs[0][...] = ctx.inputs[0]
```

While yielded, the block leaves its inputs unread and dependent blocks wait while the rest of the flowgraph and UI keep running. Always re-check your condition on every call because the scheduler may wake the block early. Returning bare `YIELD` retries on the next scheduler wake.

> [!WARNING]
> Never call `time.sleep()` inside `compute(ctx)`. Doing so blocks the shared compute thread and freezes every flowgraph and the display until the sleep finishes.

## State, Lifecycle, and Console Output

- **Module globals persist across cycles.** Use module-level variables for accumulators, lookup tables, or client handles. Top-level code runs once immediately before the first cycle after dependency reconciliation or a code reload.
- **Define `cleanup()` to release resources.** When present, `cleanup()` runs whenever the block is destroyed or reloaded. Close files, stop worker threads, and clean up sockets there.
- **Exceptions stay isolated.** An uncaught exception in `compute(ctx)` prints to the block console and skips the block without crashing the flowgraph. The block stays skipped until you reload the code or recreate it.
- **Output routes to the owning block.** Calls to `print()` and uncaught tracebacks from background threads appear in the block's console. Captured streams report as non-interactive so libraries fall back to plain formatting.
- **Do not cache `ctx` references in globals.** Tensors, attributes, and environment entries refresh each cycle, so always read them fresh from `ctx` inside `compute(ctx)`.

## Known Limitations

- All Python blocks share one interpreter process-wide, so heavy Python work in one block slows down other Python blocks. Globals remain isolated per block.
- One managed dependency environment is active across the process. Updating dependencies reloads every scheduled Python block, and updating native extensions may require restarting CyberEther.
- Output tensor geometry is fixed by the spec. Changing shapes requires reconfiguring the block.
- Deleting keys from `ctx.env` is not supported from Python.
- The Python runtime loads at startup from your configured settings. If no valid Python installation is found, Python blocks display a diagnostic and skip computing while the rest of the flowgraph runs normally.
