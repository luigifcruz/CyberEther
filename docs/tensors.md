---
title: Tensors
description: The typed, multi-device buffers that carry data through a flowgraph.
order: 330
category: Development
---

Every sample flowing through a CyberEther pipeline lives inside a **tensor**, a typed multidimensional buffer with a shape, strides, a home device, and attached metadata. Tensors are what blocks emit, what compute kernels process, and what appear inside Python blocks as zero-copy NumPy and CuPy arrays.

This guide covers tensor anatomy, zero-copy view operations, and how CyberEther shares one buffer across CPUs, GPUs, graphics surfaces, and Python without copying samples.

## Anatomy of a Tensor

Every `Tensor` combines five pieces of information:

- **Data type:** Signed integers (`I8` to `I64`), unsigned integers (`U8` to `U64`), floats (`F32`, `F64`), complex floats (`CF32`, `CF64`), and complex integers (`CI8` to `CI64`, `CU8` to `CU64`).
- **Device:** Where the underlying native buffer lives (`CPU`, `CUDA`, `Metal`, or `Vulkan`). WebGPU is a graphics-only backend and does not host tensor storage.
- **Shape and strides:** The multidimensional layout of the handle. Storage is allocated once at creation. Reshaping, slicing, and broadcasting only adjust how a handle views that storage. Fresh tensors are contiguous, and `contiguous()` reports whether a tensor remains contiguous after view operations.
- **Offset:** The starting element offset for views carved out of a larger parent buffer.
- **Attributes:** Named metadata such as `sampleRate` and `frequency` that ride alongside the tensor (see [Tensor Attributes](/docs/metadata#tensor-attributes)).

Tensors are lightweight handles over reference-counted storage. Copying a `Tensor` copies a handle rather than the underlying samples, which is why wiring blocks together never duplicates sample buffers. Storage is freed automatically when the last handle goes away.

## Creating and Accessing Tensors

Allocate a tensor with `create()`, tag its signal axes, and access elements via `at<T>()` or raw pointers via `data<T>()`:

```cpp
Tensor buffer;
JST_CHECK(buffer.create(DeviceType::CPU, DataType::CF32, {batch, samples}));
JST_CHECK(SetSignalAxes(buffer, {
    .sample = Index{1},
    .batch = Index{0},
}));

buffer.at<CF32>(0, 5) = CF32{1.0f, -1.0f};
CF32* raw = buffer.data<CF32>();
```

Element access with `at<T>()` accepts one index per dimension and respects strides, so it stays safe on sliced or permuted views. Pointer access via `data<T>()` returns the base pointer and is only safe for linear iteration when `contiguous()` is true.

### Compile-Time Typed Wrappers

When your code works with a single element type throughout, `TypedTensor<T>` locks the type at compile time so you can drop the template argument on every access:

```cpp
TypedTensor<F32> window(DeviceType::CPU, {taps});
for (U64 i = 0; i < window.size(); ++i) {
    window.at(i) = computeCoefficient(i);
}
```

The same wrapper also stores any trivially copyable C++ struct. When `T` is not a built-in numeric dtype, storage falls back to bytes with the final dimension scaled by `sizeof(T)`, which is how render components store vertex and instance buffers:

```cpp
struct Vertex {
    F32 x;
    F32 y;
};

TypedTensor<Vertex> vertices(DeviceType::CPU, {count});
vertices.at(0) = {0.0f, 1.0f};
```

### Wrapping External Memory

You can wrap pre-allocated memory owned by an external library or hardware driver without copying it:

```cpp
std::vector<CF32> samples(2048);

Tensor wrapped;
JST_CHECK(wrapped.create(samples.data(), DeviceType::CPU, DataType::CF32, {2048}));
```

The tensor borrows that pointer without taking ownership, so the backing allocation must outlive every handle and downstream consumer.

> [!WARNING]
> Calling `clone()` does **not** copy sample data. It creates a new handle with independent shape, strides, and attributes pointing to the **same underlying storage**. Use `clone()` before reshaping a view without mutating the original handle. When you need an independent deep copy, allocate a new tensor and call `copyFrom()`.

## Zero-Copy Views

Every structural transformation mutates the handle in place and returns a `Result` without touching the underlying sample bytes. To preserve the original layout, call `clone()` first and transform the clone:

| Operation | Method | What It Does |
|---|---|---|
| Add or remove unit axis | `expandDims(axis)`, `squeezeDims(axis)` | Changes tensor rank without moving data. |
| Change layout | `reshape(shape)`, `permute(axes)` | Reinterprets dimensions or reorders axes. |
| Virtual repeat | `broadcastTo(shape)` | Repeats data virtually using zero strides. |
| Extract sub-range | `slice(tokens)` | Slices per axis using an index, `{start, stop}`, or `{start, stop, step}` (mirroring Python slicing). |

Blocks like Slice and Permutation rely on these views, which means downstream modules can receive strided, non-contiguous inputs. Structural blocks remap positional metadata (`sampleAxis`, `batchAxis`, and `channelAxis`) or drop roles when a reshape makes them ambiguous.

To prevent silent corruption, the framework enforces contiguity at module creation: any module that has not declared the `DISCONTIGUOUS` taint (see [Module Lifecycle](/docs/module-contract#module-lifecycle)) rejects non-contiguous inputs automatically.

## One Tensor, Many Devices

A tensor's backing storage is not locked to one device. Instead, it holds an on-demand table of per-device buffers. Querying `hasDevice(device)` or constructing `Tensor(device, source)` maps existing storage onto another device with **zero copies whenever the hardware and drivers allow**. Call `device()` to inspect the active view's device and `nativeDevice()` to check where the buffer was originally allocated.

### Device Mapping Matrix

Mappings are directional. Rows indicate where the storage lives natively, columns indicate the requested view device, and `hasDevice()` reflects this exact table at runtime:

| Native Storage | CPU View | CUDA View | Metal View | Vulkan View |
|---|---|---|---|---|
| **CPU** | Native | Yes (pinned pages) | Yes (shared wrap) | No |
| **CUDA** | Conditional (unified only) | Native | No | Conditional (external memory) |
| **Metal** | Yes (always) | No | Native | No |
| **Vulkan** | Conditional (host-visible only) | Yes (external memory) | No | Native |

How each conditional mapping behaves:

- **From CPU to CUDA:** Pins the host pages with the CUDA driver. Requires page-aligned memory (framework allocations always qualify, while wrapped external pointers may not).
- **From CPU to Metal:** Wraps the same page-aligned memory as a Metal shared-mode buffer.
- **From CUDA to CPU:** Succeeds only when the CUDA tensor was allocated with `{.hostAccessible = true}` (managed memory). Standard device-only CUDA allocations have no CPU view.
- **From CUDA to Vulkan:** Exports the buffer via an external memory handle. Requires a default device-local CUDA allocation (not `hostAccessible`) and driver support on both sides. Unavailable on Windows.
- **From Vulkan to CUDA:** Imports the Vulkan buffer via external memory, with the same driver requirements and Windows exclusion.
- **From Vulkan to CPU:** Uses the mapped host pointer, requiring a host-visible buffer (true on unified-memory hardware and `hostAccessible` allocations, false for discrete device-local GPU buffers).
- **From Metal to CPU:** Always available because Metal buffers allocate in shared mode.

Because WebGPU is graphics-only and has no tensor buffer backend, the render layer feeds it through explicit uploads. Pairs marked No bridge by staging through a mutually mappable device (typically CPU) using `copyFrom()`.

### Sharing With the Host

Mapping a tensor to another device takes one constructor call, and both handles alias the same bytes immediately:

```cpp
Tensor host;
JST_CHECK(host.create(DeviceType::CPU, DataType::CF32, {2048}));

Tensor device(DeviceType::CUDA, host);
```

A CUDA kernel writing to `device` writes straight into the memory `host` reads, with no transfer step. In the reverse direction, a CPU view of a CUDA tensor requires a host-accessible allocation. Calling `hasDevice(device)` tests whether a zero-copy mapping is supported before you commit to it.

### Sharing Compute With Graphics

The same constructor bridges compute kernels and GPU rendering. A CUDA view over a Vulkan tensor uses external memory handles so a compute kernel can populate a render surface directly with no readback or staging copy:

```cpp
Tensor surface;
JST_CHECK(surface.create(DeviceType::Vulkan, DataType::F32, {height, width}));

Tensor compute(DeviceType::CUDA, surface);
```

Always check `surface.hasDevice(DeviceType::CUDA)` first to handle platforms or drivers without external memory support gracefully.

### Unified Memory Allocations

On unified-memory hardware, you can request CPU visibility up front during allocation:

```cpp
Tensor unified;
JST_CHECK(unified.create(DeviceType::CUDA, DataType::CF32, {2048}, {.hostAccessible = true}));
```

### Explicit Copies

When you genuinely want an independent copy of the samples, transfer them explicitly with `copyFrom()`:

```cpp
Tensor upload;
JST_CHECK(upload.create(DeviceType::CUDA, DataType::CF32, host.shape()));
JST_CHECK(upload.copyFrom(host));
```

Note that `copyFrom()` uses the same mapping layer underneath: the destination must be contiguous, shapes and sizes must match, and the destination must be able to materialize a view on the source's device where the copy executes. When a direct mapping does not exist for a device pair, `copyFrom()` fails cleanly, and you stage through a common device such as the CPU. Because mappings either alias directly or fail cleanly, every real data copy in a flowgraph is visible as an explicit `copyFrom()`.

### Crossing Devices Inside Modules

Declare the `CROSS_DEVICE` taint whenever your module kernel can consume inputs living on another device. Without `CROSS_DEVICE`, the framework rejects cross-device inputs at module creation, and the owning block must bridge explicitly by creating a device view of the source tensor. A single block can emit CPU and CUDA outputs side by side, letting heterogeneous graphs run without hidden CPU staging copies.

## Zero-Copy Python Interop

Tensors cross into the [Python Block](/docs/python-block) with zero copies in both directions:

- Each **CPU tensor** arrives in `compute(ctx)` as a NumPy array backed by a memoryview over the tensor's memory, preserving strides for non-contiguous views.
- Each **CUDA tensor** arrives as a CuPy array wrapping the raw unowned device pointer with strides intact.
- Inputs are read-only (enforced by NumPy on CPU and by a read-only wrapper on CuPy), while outputs are writable in place.

```python
import cupy as cp

def compute(ctx):
    x = ctx.inputs[0]                    # cupy.ndarray over the tensor's device memory
    ctx.outputs[0][...] = cp.fft.fft(x)  # writes straight into the downstream tensor
    cp.cuda.Stream.null.synchronize()
```

There is no serialization boundary between C++ and Python for bulk sample data, only for the lightweight metadata documented in [Type Conversion](/docs/metadata#type-conversion).

## Attributes Ride Along

Attributes attach directly to tensors and follow them across links, views, and devices. Producers attach values with `setAttribute()`, pass-through modules copy them with `propagateAttributes()`, and `setDerivedAttribute()` registers a callback evaluated on read for values derived from live state.

Any structural operation that changes axes must remap `sampleAxis`, `batchAxis`, and `channelAxis` to the new numbering or drop roles that no longer apply. Never forward stale axis indices onto a reshaped tensor. See [Signal Axes](/docs/metadata#signal-axes) and [Tensor Attributes](/docs/metadata#tensor-attributes) for the full contract.

## Best Practices

- **Allocate in `create()`, reuse every cycle.** Output tensors are created once and overwritten each cycle. Allocating inside `computeSubmit()` hurts performance and leaves downstream consumers holding the old buffer.
- **Publish outputs before returning from `create()`.** Downstream blocks only see an output after `produced()` registers it, locking its shape and dtype until a `RECREATE`.
- **Lean on views.** Slicing a channel, squeezing an axis, or permuting dimensions costs zero copies per cycle. Remember that `clone()` shares storage, so pair a fresh tensor with `copyFrom()` whenever downstream needs independent or contiguous memory.
- **Check `contiguous()` before walking `data<T>()`.** Linear pointer arithmetic breaks as soon as an upstream Slice block appears. Use `at<T>()` or respect strides explicitly.
- **Keep numeric widths consistent.** CyberEther's dtype ladder mirrors NumPy (`CF32` matches `complex64`), so sticking to consistent widths across tensors, attributes, and environment values avoids unnecessary conversions (see [Type Conversion](/docs/metadata#type-conversion)).
