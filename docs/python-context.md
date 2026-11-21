---
title: Python Context API
description: Work with zero-copy NumPy and CuPy views, CUDA synchronization, tensor attributes, ctx.env, ctx.metrics, and background threads.
order: 232
category: Usage
parent: python-block
---

Every cycle, CyberEther calls your [Python Block](/docs/python-block) script with a `ctx` object that exposes tensors, tensor attributes, the shared flowgraph environment, and block metrics.

## Working With Tensors

Both `ctx.inputs` and `ctx.outputs` are indexed by port number. Each CPU tensor arrives as a zero-copy NumPy array, and each CUDA tensor arrives as a zero-copy CuPy array:

```python
def compute(ctx):
    x = ctx.inputs[0]                  # numpy.ndarray (CPU) or cupy.ndarray (CUDA)
    ctx.outputs[0][...] = x[::2] * 0.5
```

### Working With CUDA Tensors

Using GPU tensors requires [CuPy](https://cupy.dev) in the active Python runtime, while NumPy handles CPU tensors. Neither library is imported until a tensor on that device actually exists, so CPU-only setups never need CuPy installed.

Input tensors are guaranteed to be ready when `compute(ctx)` starts, though CuPy kernel launches are asynchronous. Always synchronize before returning so downstream blocks read completed results:

```python
import cupy as cp

def compute(ctx):
    ctx.outputs[0][...] = cp.fft.fft(ctx.inputs[0])
    cp.cuda.Stream.null.synchronize()
```

If you launch kernels on custom CuPy streams, synchronize those streams before returning as well.

## Tensor Attributes

Tensors carry named metadata such as `sampleRate` and `frequency`. Each port exposes its own attribute mapping:

- **Input attributes (`ctx.input_attrs[i]`):** Read-only mapping refreshed at the start of every cycle, including propagated and derived attributes from upstream blocks.
- **Output attributes (`ctx.output_attrs[i]`):** Writable dictionary published when `compute(ctx)` returns, making updates visible to downstream blocks in the same cycle and to pin tooltips in the UI. Axes declared in the output tensor spec are excluded (see [Declaring Signal Axes](/docs/python-block#declaring-signal-axes)).

```python
def compute(ctx):
    rate = ctx.input_attrs[0].get("sampleRate", 0.0)

    ctx.outputs[0][...] = ctx.inputs[0][::2]

    ctx.output_attrs[0].update(ctx.input_attrs[0])
    ctx.output_attrs[0]["sampleRate"] = rate / 2.0
    ctx.output_attrs[0]["decimation"] = 2
```

In-place edits to nested containers (like `ctx.output_attrs[0]["meta"]["stage"] = 2`) are detected and published automatically, while untouched attributes remain unchanged.

The Python block does not copy input attributes to outputs automatically, so explicitly forward every attribute that still applies. Signal outputs must preserve or remap `sampleAxis`, `batchAxis`, and `channelAxis`, and any newly assigned axis index must use `numpy.uint64` to satisfy the `Index` (`U64`) contract. Every assignment stores the exact type you write (matching C++ `setAttribute`), so `numpy.float32` stores `F32` whereas a plain Python `float` stores `F64`.

## Flowgraph Environment

The `ctx.env` mapping connects your script to the flowgraph environment, a graph-wide timestamped key-value store also visible in the **Environment** tile and to C++ blocks. Values refresh before `compute(ctx)` runs, and edits publish atomically when `compute(ctx)` returns:

```python
def compute(ctx):
    gain = ctx.env.get("station", {}).get("gain", 1.0)
    ctx.outputs[0][...] = ctx.inputs[0] * gain

    ctx.env["status"] = {
        "peak": abs(ctx.outputs[0]).max(),
    }
```

Key behaviors to keep in mind:

- **Top-level entries must be dictionaries.** Assigning a bare scalar like `ctx.env["gain"] = 3.0` is rejected with a console warning. Wrap values inside a dict.
- **Null values round-trip cleanly.** Writing `None` stores null, renders as `null` in the Environment tile, and reads back as `None`.
- **Exact numeric widths are preserved.** When NumPy is available, numbers (including complex values) read and write at their exact stored widths, and typed vectors arrive as read-only NumPy arrays. Without NumPy, reads fall back to standard Python types. See the full [Type Conversion](/docs/metadata#type-conversion) table in the Metadata guide.
- **Nested edits are tracked.** Both `ctx.env["k"] = {...}` and `ctx.env["k"]["field"] = value` publish changes, including dicts nested inside sequences. Sequences read back as immutable tuples, so replace them through their parent dict.
- **Invalid writes roll back.** If a value cannot be converted or published, the local entry reverts to the flowgraph's canonical state rather than silently diverging.
- **Deletions are not supported.** Running `del ctx.env["k"]` reverts on the next cycle. Clear keys from C++ or the UI instead.
- **Publishes are cycle-atomic.** Downstream blocks running later in the same cycle see your updates, with no partial visibility mid-compute.
- **Environment state is runtime-only.** Values live in memory while the flowgraph runs and are not saved inside the `.yml` file. Blocks that need startup values should republish them when the flowgraph loads.
- **New keys trigger incomplete-block retries.** When a new key appears (or is cleared), blocks waiting in the incomplete state automatically recreate so they can pick it up. Updating an existing key in place never triggers recreation, keeping per-cycle status updates cheap.

Refresh checks are version-gated per key, and publishing only inspects keys your code touched. Even an environment with thousands of keys costs virtually nothing per cycle while idle, with only initial population proportional to total key count. If you ingest an external feed that resends full snapshots, diff against the previous snapshot and assign only changed keys.

## Block Metrics

The `ctx.metrics` mapping provides read-only access to metrics published by other blocks, keyed by block name and metric name. When you only need the current value, call `get_value()`:

```python
def compute(ctx):
    progress = ctx.metrics.get_value("file_reader", "progress", default=0.0)
    throughput = ctx.metrics.get_value("websocket_receiver", "throughput")
```

Passing a `default` returns that value instead of `None` when the block or metric is missing or still priming.

Each metric entry is also a dictionary containing both its value and UI metadata:

```python
metric = ctx.metrics["file_reader"].get("progress")
if metric is not None:
    metric["value"]   # raw metric value
    metric["format"]  # descriptor dictionary
    metric["label"]   # display label
    metric["help"]    # description
```

### How Subscriptions Work

Metric fetching is subscription-based so unsubscribed blocks incur zero overhead. The very first time your script touches a block name (via `get_value()`, `[]`, or `.get()`), CyberEther subscribes to that block and returns an empty result for that cycle. Starting on the next cycle, that block's metrics refresh automatically before `compute(ctx)` runs. Because of this one-cycle priming delay, always pass a default to `get_value()` or guard against `None`.

To subscribe to every block dynamically without hard-coding names, call `ctx.metrics.subscribe_all()`:

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

Additional metric details:

- Subscribing to a block that does not exist yet returns an empty mapping and begins populating automatically if that block is added later.
- Progress-bar metrics arrive as a `(label, fraction)` tuple in `"value"`. Some blocks publish formatted display strings (such as `"12.3 MB/s"`) rather than raw numbers, so check the type you receive.
- Metrics whose C++ type cannot be represented in Python are omitted.
- Writes to `ctx.metrics` are ignored by the flowgraph and overwritten on the next refresh.

## Integrating Blocking External Services

Because the scheduler is synchronous, `compute(ctx)` must never block on sockets, HTTP requests, or queues. Run blocking I/O in a background thread and drain results from a `queue.Queue` inside `compute(ctx)`:

```python
import queue
import threading

_updates = queue.Queue()
_stop = threading.Event()

def _worker():
    while not _stop.is_set():
        snapshot = fetch_next_snapshot()  # blocking network call
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

Background threads run freely between compute cycles because CyberEther releases the Python interpreter lock between steps. Worker threads may call `print()` safely, though they must never touch `ctx` directly. Only `compute(ctx)` is allowed to read or write `ctx`.

> [!WARNING]
> Some compiled third-party client libraries hold the Python interpreter lock while blocking on network calls, which stalls every Python block in the process even from a background thread. If the timing display freezes as soon as a worker connects, switch to the library's async API on an event loop inside the worker thread, poll over plain HTTP with the standard library, or isolate the client in a helper subprocess.
