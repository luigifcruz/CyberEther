---
title: Python Block
description: Run custom Python compute code inside a flowgraph.
order: 40
category: Usage
---

The Python block executes user-provided Python code on every compute cycle, with zero-copy access to the tensors flowing through the graph, the tensor metadata attached to them, and the flowgraph environment. It is the fastest way to prototype a processing step, glue two blocks together with custom logic, or bring the NumPy, CuPy, and SciPy ecosystem into a flowgraph without writing a native block.

## Quick Start

Add a Python block to a flowgraph (it requires a [Python runtime](#choosing-a-python-runtime)), give it one input and one output, and define a `compute(ctx)` function:

```python
def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0] * 2.0
```

The code is compiled when the block is created and `compute(ctx)` runs once per compute cycle. Editing the code in the node reloads it in place. Changing the input or output counts recreates the block.

## Choosing a Python Runtime

CyberEther does not ship with its own Python. The block runs on a Python installation already on your system, the same one you use from the terminal, so every package installed there (NumPy, SciPy, CuPy, Astropy, and so on) is available to `compute`. If you already have an environment set up, you can point CyberEther at it and use it directly inside a flowgraph.

To pick which installation is used, open **Settings**, select the **Runtime** tab, and use the **Python Runtime** selector:

- **Auto** (the default) scans the system and uses the first working Python it finds. The scan covers the executables on your `PATH` as well as active `venv`, Conda, and pyenv environments, so launching CyberEther from a terminal with your environment activated is enough for Auto to pick it up.
- The dropdown also lists every installation the scan found, labeled with its version and location, such as `Python 3.12.4 (/opt/miniconda3/bin/python)`. Pick the one that has the packages you need.
- **Custom Path** accepts the path to any `python` executable the scan did not find (a leading `~` works), typed directly or chosen with **Browse File**.

A badge next to the selector reports whether the choice is usable: **Valid File** means a matching Python library was located, **Invalid** means it was not. The selection is saved and applies after restarting CyberEther.

A useful rule of thumb: if `python -c "import numpy"` works in your terminal, selecting that same Python here makes the import work in the block too. Conversely, if an import fails inside the block, check which runtime is selected before reinstalling packages. The block may simply be running a different Python than your terminal.

## Block Configuration

| Field | Meaning |
|---|---|
| **Code** | Python source defining `compute(ctx)`. |
| **Input Count** | Number of input ports (`input0`, `input1`, ...). |
| **Output Count** | Number of output ports (`output0`, `output1`, ...). |
| **Output Tensor Specs** | Per-output shape, data type, and device. |

Each output tensor is allocated by the block from its spec. The Python code cannot change an output's shape, dtype, or device at runtime. Supported spec dtypes are `F32`, `CF32`, `F64`, `CF64`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, and `U64`. Supported devices are `cpu` and `cuda`. Blocks with zero inputs (sources) and zero outputs (sinks) are both valid.

## Working With Tensors

The `ctx.inputs` and `ctx.outputs` mappings are keyed by port index. CPU tensors arrive as NumPy arrays and CUDA tensors as CuPy arrays, both zero-copy views over the tensor memory:

```python
def compute(ctx):
    x = ctx.inputs[0]                  # numpy.ndarray (CPU) or cupy.ndarray (CUDA)
    ctx.outputs[0][...] = x[::2] * 0.5
```

Rules that matter:

- **Write outputs in place.** `ctx.outputs[0][...] = value` writes to the flowgraph. Rebinding the name (`out = value`) writes nothing.
- **Inputs are read-only.** CPU inputs are enforced by NumPy, while CUDA inputs are wrapped in a read-only array type. Copy before mutating.
- **Non-contiguous inputs work.** Strided views produced by blocks like `slice` or `permutation` map to properly strided arrays, with no copies and no restrictions.
- **Devices can be mixed.** A single block can read CPU and CUDA inputs and produce outputs on either device, independent of the block's own device.

### CUDA Notes

CUDA tensors require [CuPy](https://cupy.dev) in the Python runtime's environment. NumPy covers CPU tensors. Neither is imported until a tensor of that kind actually exists, so CPU-only systems never need CuPy installed.

Synchronization contract: inputs are guaranteed complete when `compute` starts, but CuPy launches are asynchronous. Synchronize before returning so that downstream blocks see finished writes:

```python
import cupy as cp

def compute(ctx):
    ctx.outputs[0][...] = cp.fft.fft(ctx.inputs[0])
    cp.cuda.Stream.null.synchronize()
```

Work submitted on custom CuPy streams is likewise the user's responsibility to synchronize before `compute` returns.

## Tensor Attributes

Tensors carry named metadata such as `sampleRate` and `frequency`. The block exposes them per port:

- `ctx.input_attrs[i]`: read-only mapping of the input tensor's attributes, including values inherited through upstream propagation and derived attributes, refreshed at the start of every cycle.
- `ctx.output_attrs[i]`: writable dict for the output tensor. Writes are published when `compute` returns and become visible to downstream blocks in the same cycle, and to pin tooltips in the UI.

```python
def compute(ctx):
    rate = ctx.input_attrs[0].get("sampleRate", 0.0)

    ctx.outputs[0][...] = ctx.inputs[0][::2]

    ctx.output_attrs[0]["sampleRate"] = rate / 2.0
    ctx.output_attrs[0]["decimation"] = 2
```

Editing a container-valued attribute in place (for example `ctx.output_attrs[0]["meta"]["stage"] = 2`) is detected and published as well. Attributes the block does not touch are left as-is. The block does not automatically propagate input attributes to outputs, so copy the ones you want.

## Flowgraph Environment

The `ctx.env` mapping mirrors the flowgraph environment: a graph-wide, timestamped key-value store also visible in the Environment window and to other blocks. Values refresh at the start of each cycle and writes are published when `compute` returns.

```python
def compute(ctx):
    gain = ctx.env.get("station", {}).get("gain", 1.0)
    ctx.outputs[0][...] = ctx.inputs[0] * gain

    ctx.env["status"] = {
        "peak": float(abs(ctx.outputs[0]).max()),
    }
```

Semantics:

- **Top-level values must be mappings.** `ctx.env["gain"] = 3.0` is rejected with a console warning, so wrap scalars in a dict.
- **Complex values work.** A Python `complex` is stored as a 64-bit complex (CF64). Writes over an entry seeded as CF32 from the C++ side keep it CF32. Both kinds read back as Python `complex`, in the environment and in tensor attributes alike.
- **Nested edits publish.** Both `ctx.env["k"] = {...}` and `ctx.env["k"]["field"] = value` are tracked, including dicts nested inside sequences. Sequences read back as immutable tuples. Replace them through their parent dict.
- **Rejected writes are rolled back.** If a value cannot be converted or published, the local entry is restored to the flowgraph's canonical state instead of silently diverging.
- **Deletions are not supported.** `del ctx.env["k"]` is reverted on the next cycle. Clear keys from the C++ side or the UI instead.
- **Writes are cycle-atomic.** Downstream blocks that run later in the same cycle see the updated values, and there is no mid-compute visibility.
- **The environment persists with the flowgraph.** Keys written by Python are serialized when the flowgraph is saved.
- **New keys retry incomplete blocks.** When a key first becomes visible (or is cleared), blocks sitting in the incomplete state are destroyed and recreated so they can pick the value up. This is how blocks that gate their creation on server-provided values start once the connection delivers them. In-place updates to an existing key do not trigger retries, so per-cycle status writes stay cheap.

The refresh path is epoch-gated and the publish path only examines keys the code actually touched, so a large environment (thousands of keys) costs nothing per cycle while idle. The expensive moments are the initial population and the cycle after any change, both proportional to the total key count. When consuming an external feed that resends full snapshots, diff against the previous snapshot and assign only changed keys.

## Block Metrics

The `ctx.metrics` mapping gives read-only access to metrics published by other blocks in the flowgraph, keyed by block name and metric name:

```python
def compute(ctx):
    progress = ctx.metrics["file_reader"].get("progress")
    throughput = ctx.metrics["websocket"].get("throughput")
```

Access is subscription-based. The first read of a block's name registers interest and returns an empty mapping. From the next cycle on, that block's metrics are refreshed at the start of every cycle. Because of the one-cycle priming delay, always read metric values with `.get()` and a sensible default.

To subscribe to every block without hard-coding names, call `ctx.metrics.subscribe_all()`. The dictionary is populated on the next cycle, then you can iterate the currently visible metrics:

```python
_primed = False

def compute(ctx):
    global _primed
    ctx.metrics.subscribe_all()
    if not _primed:
        _primed = True
        return

    for block, metrics in ctx.metrics.items():
        print(block, metrics)
```

Details worth knowing:

- Only subscribed blocks are evaluated, so unrelated metrics cost nothing.
- A subscription to a block that does not exist (yet) yields an empty mapping and starts producing values if the block appears later.
- Metrics with `private-` formats (internal timing and diagnostics) are hidden.
- Values arrive with their native types when possible. Progress-bar style metrics come through as a `(label, fraction)` tuple. Note that some blocks publish display-formatted strings (for example `"12.3 MB/s"`) rather than raw numbers, so check the shape of what you receive.
- The mapping is read-only in spirit: writes to it are ignored by the flowgraph and overwritten on refresh.

## Type Conversion

Values crossing between C++ and Python convert as follows.

Reading (C++ to Python):

| C++ value | Python |
|---|---|
| `F32`, `F64` | `float` |
| `I8` to `I64`, `U8` to `U64` | `int` |
| `bool` | `bool` |
| `std::string` | `str` |
| `std::vector<F32/F64/U64>` | `tuple` |
| `Parser::Map` | `dict` (recursive) |
| `Parser::Sequence` | `tuple` (recursive) |

Writing (Python to C++): `bool`, `int`, `float`, `str`, `dict`, and `list`/`tuple` are supported. If the key already holds a value, the write is **coerced to the existing type**: an `F32 sampleRate` stays `F32`, integer targets are range-checked exactly, and floats only land in integer slots when they are integral (writing `1.5` over an integer is rejected with a warning rather than truncated). New keys default to `bool`, `I64` (`U64` above the signed range), `F64`, and `str`. New homogeneous lists become `vector<F64>` (floats) or `vector<U64>` (non-negative integers), and anything mixed becomes a generic sequence.

Complex scalars (`CF32`/`CF64`) and framework types (`Range`, `Extent2D`, `Tensor`, device enums) do not cross the bridge in either direction: they are skipped on read and rejected with a console warning on write.

## State And Lifecycle

- **Globals persist across cycles.** Module-level variables survive between `compute` calls. Use them for accumulators, precomputed tables, or open connections. Module-level code runs once, at block creation or code reload.
- **Optional `cleanup()` hook.** If defined, it runs when the block is destroyed or the code is reloaded. Close files, stop threads, and release resources there.
- **Errors do not stop the flowgraph.** Exceptions raised by `compute` are captured in the block console. The failing cycle is skipped, and the block remains skipped until the code is reloaded or the block is recreated. Output written to `print()` also appears in the block console.
- **References go stale across cycles.** Arrays, attribute values, and environment entries are refreshed each cycle, so fetch them from `ctx` inside `compute` rather than caching them in globals.

## Feeding Data From External Services

The `compute` function must not block: the scheduler is synchronous, and a stalled `compute` stalls the whole flowgraph while holding the interpreter. For subscriptions, sockets, or polling loops, run the connection in a background thread and drain it from `compute`:

```python
import queue
import threading

_updates = queue.Queue()
_stop = threading.Event()

def _worker():
    while not _stop.is_set():
        snapshot = fetch_next_snapshot()   # blocking network call
        _updates.put(snapshot)

_thread = threading.Thread(target=_worker, daemon=True)
_thread.start()

def compute(ctx):
    while not _updates.empty():
        snapshot = _updates.get_nowait()
        for key, value in snapshot.items():
            ctx.env[key] = value

def cleanup():
    _stop.set()
    _thread.join(timeout=2.0)
```

Background threads run freely while the flowgraph computes (the interpreter lock is released between cycles), but they must never touch `ctx`. Only `compute` may read or write it. Report worker errors by queueing them and printing from `compute`, since console capture wraps compute calls only.

## Limitations

- One Python interpreter is shared by all Python blocks in the process, so heavy computation in one block delays the others. Blocks do get isolated globals, so one block's variables are not visible to another.
- Output tensor geometry is fixed by the spec. Reshaping on the fly requires reconfiguring the block.
- Environment deletions and complex-valued metadata are not supported (see above).
- The runtime loads at startup from the installation selected in the settings (see [Choosing a Python Runtime](#choosing-a-python-runtime)). If no usable Python is found, the block reports it in its diagnostic and skips computing without affecting the rest of the flowgraph.
