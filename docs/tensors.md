---
title: Tensors
description: The typed, multi-device buffers that carry data through a flowgraph.
order: 80
category: Development
---

Every sample that moves through a flowgraph lives in a tensor: a typed n-dimensional buffer with a shape, strides, a device, and attached metadata. Tensors are what block outputs produce, what module kernels read and write, and what crosses into Python as NumPy and CuPy arrays. This page covers their anatomy, the zero-copy view operations, and the part that makes CyberEther graphs composable across hardware, namely how one tensor interoperates across devices and languages without copying samples.

## Anatomy

A tensor is described by five things:

- A **data type** from the numeric ladder: `I8` to `I64`, `U8` to `U64`, `F32`, `F64`, the complex floats `CF32` and `CF64`, and the complex integers `CI8` to `CI64` and `CU8` to `CU64`.
- A **device** where its native buffer lives: `CPU`, `CUDA`, `Metal`, `Vulkan`, or `WebGPU`.
- A **shape** and **strides** describing the n-dimensional layout. A freshly created tensor is contiguous, and `contiguous()` reports whether it still is after view operations.
- An **offset** for views that start inside a parent buffer.
- **Attributes**, the named metadata such as `sampleRate` that travels with the tensor, covered in [Metadata](/docs/metadata#tensor-attributes).

Tensors are cheap handles over shared storage. Copying a `Tensor` object copies a reference, not samples, which is why connecting blocks and passing tensors through links never duplicates data. The underlying storage is freed when the last handle drops.

## Creating And Accessing

```cpp
Tensor buffer;
JST_CHECK(buffer.create(DeviceType::CPU, DataType::CF32, {batch, samples}));

buffer.at<CF32>(0, 5) = CF32{1.0f, -1.0f};
CF32* raw = buffer.data<CF32>();
```

Element access with `at<T>()` takes one index per dimension and honors strides, so it stays correct on sliced and permuted views. Raw access with `data<T>()` returns the base pointer, which is only safe to walk linearly when `contiguous()` holds.

For code that works with one element type throughout, `TypedTensor<T>` fixes the dtype at compile time and drops the template argument from every access:

```cpp
TypedTensor<F32> window(DeviceType::CPU, {taps});
for (U64 i = 0; i < window.size(); ++i) {
    window.at(i) = computeCoefficient(i);
}
```

The same wrapper also stores arbitrary trivially copyable structs. When the element type is not a framework dtype, the storage falls back to bytes with the last axis scaled by the element size, which is how the render components keep vertex and instance data in tensors:

```cpp
struct Vertex {
    F32 x;
    F32 y;
};

TypedTensor<Vertex> vertices(DeviceType::CPU, {count});
vertices.at(0) = {0.0f, 1.0f};
```

A tensor can also wrap pre-allocated memory it does not own by creating from an existing pointer:

```cpp
std::vector<CF32> samples(2048);

Tensor wrapped;
JST_CHECK(wrapped.create(samples.data(), DeviceType::CPU, DataType::CF32, {2048}));
```

The tensor borrows the memory rather than taking ownership, so the allocation must outlive every handle and every downstream consumer. This is the bridge for feeding buffers owned by external libraries or drivers into a flowgraph without a copy.

One method deserves a warning label: `clone()` does **not** copy data. It produces a new handle with its own shape, strides, and attributes over the **same storage**, which is exactly what you want before applying view operations without disturbing the original. A true deep copy is the explicit two-step of creating a fresh tensor and transferring into it with `copyFrom`.

## Views

The reshaping operations never copy data. Each one mutates the handle it is called on, returning a `Result`, so the tensor becomes a view over the same storage with a new layout. To keep the original layout around, `clone()` a handle first (cheap, shares storage) and reshape the clone:

- Rank changes with `expandDims(axis)` and `squeezeDims(axis)`.
- Layout changes with `reshape(shape)` and `permute(axes)`.
- Size changes with `broadcastTo(shape)`, which repeats data virtually through zero strides.
- Sub-ranges with `slice(tokens)`, where each token is an index, a `{start, stop}` pair, or a `{start, stop, step}` triple per axis, mirroring Python slicing.

Views are how the slice and permutation blocks work, and they are the reason strides exist in the module contract: a downstream module receives whatever layout the view produced. The framework enforces the contract at module creation. A module that has not declared the `DISCONTIGUOUS` taint described in [Module Lifecycle](/docs/blocks-and-modules#module-lifecycle) rejects non-contiguous input with an error rather than silently misreading it, so kernels only ever see layouts they claimed to handle.

## One Tensor, Many Devices

This is the interoperability core. A tensor's storage is not a single buffer but a table of per-device buffers grown on demand. Asking `hasDevice(device)`, or constructing `Tensor(device, source)`, maps the existing storage onto another device, and the mapping is **zero-copy whenever the platform allows**. The `device()` accessor reports the buffer a consumer is using, and `nativeDevice()` reports where the storage originally lived.

### Sharing With The Host

Mapping a tensor onto another device is one constructor call, and both views alias the same bytes from then on. A CUDA view of a CPU tensor pins the host pages with the driver and computes against them directly:

```cpp
Tensor host;
JST_CHECK(host.create(DeviceType::CPU, DataType::CF32, {2048}));

Tensor device(DeviceType::CUDA, host);
```

A kernel writing through `device` is writing the memory `host` reads, so a CPU consumer sees the results without any transfer. The reverse direction works the same way, and `hasDevice(device)` performs the identical mapping as a query, returning false when the platform cannot share the memory.

### Sharing With Graphics

The same one-liner bridges compute and graphics. A CUDA view of a Vulkan tensor goes through external memory handles, so a kernel can fill the exact buffer a render surface draws, with no readback and no staging:

```cpp
Tensor surface;
JST_CHECK(surface.create(DeviceType::Vulkan, DataType::F32, {height, width}));

Tensor compute(DeviceType::CUDA, surface);
```

Availability follows the drivers: the import requires external memory support from both sides, and `hasDevice(DeviceType::CUDA)` on the Vulkan tensor is the portable way to test for it before committing to the zero-copy path.

### Unified Memory

Device allocations can also request CPU visibility up front through the buffer configuration, which is the natural shape on unified-memory hardware:

```cpp
Tensor unified;
JST_CHECK(unified.create(DeviceType::CUDA, DataType::CF32, {2048}, {.hostAccessible = true}));
```

### Explicit Transfers

When an independent copy of the data is genuinely wanted, rather than another view of it, the transfer is explicit:

```cpp
Tensor upload;
JST_CHECK(upload.create(DeviceType::CUDA, DataType::CF32, host.shape()));
JST_CHECK(upload.copyFrom(host));
```

Note that `copyFrom` rides the same mapping machinery as everything else: the destination must be contiguous, the sizes must match, and the destination storage must be able to materialize a buffer on the source's device, where the actual copy then happens. On platforms where that mapping does not exist for a given device pair, `copyFrom` fails cleanly too, and bridging requires staging through a device both sides can map, in practice the CPU. Since mappings either succeed as aliases or fail cleanly, every actual transfer in a flowgraph is a visible `copyFrom` in somebody's code.

### Crossing Devices In Modules

For module authors the practical rule is: declare the `CROSS_DEVICE` taint when the kernel can consume input living on another device. Without it, input on a different device is rejected at module creation, and the block is responsible for bridging explicitly by constructing a device view of the source tensor. A single block can produce CPU and CUDA outputs side by side, and mixed graphs work without staging copies through the CPU.

## Python Without Copies

The same buffers cross into the Python block with zero copies in both directions:

- A CPU tensor appears in `compute(ctx)` as a NumPy array built over a memoryview of the tensor's own memory, strides included, so non-contiguous views arrive as properly strided arrays.
- A CUDA tensor appears as a CuPy array wrapping the raw device pointer through unowned memory, again with strides intact.
- Inputs arrive read-only (enforced by NumPy for CPU tensors and by a read-only array wrapper for CuPy), while outputs are writable in place.

The consequence is that Python participates in the same aliasing as everything else:

```python
import cupy as cp

def compute(ctx):
    x = ctx.inputs[0]                    # cupy.ndarray over the tensor's device memory
    ctx.outputs[0][...] = cp.fft.fft(x)  # lands directly in the downstream tensor
    cp.cuda.Stream.null.synchronize()
```

The write on the second line goes straight into the buffer the next block computes on. There is no serialization boundary between C++ and Python for sample data, only for the small typed metadata described in [Type Conversion](/docs/python-block#type-conversion).

## Attributes Ride Along

Attributes attach to the tensor and follow it through links, views, and devices. Producers set them with `setAttribute`, pass-through blocks forward them with `propagateAttributes`, and `setDerivedAttribute` registers a callable evaluated on read for values computed from live state. The conventions and hints live in [Metadata](/docs/metadata#tensor-attributes).

## Hints

- **Allocate in `create()`, reuse across cycles.** Output tensors are created once when the module is created and rewritten every cycle. Allocating inside the compute path is both slow and invisible to consumers holding the old buffer.
- **Publish before returning.** A module output only exists for downstream blocks once `produced()` registered the tensor, and the shape and dtype are fixed from that point. Geometry changes go through `RECREATE`.
- **Let views do the work.** Selecting a channel, dropping a dimension, or transposing is a view, and chaining views costs nothing per cycle. Remember that `clone()` only forks the layout, so reach for a fresh tensor plus `copyFrom()` when downstream genuinely needs independent or contiguous storage.
- **Check `contiguous()` before pointer arithmetic.** Any code that walks `data<T>()` linearly is a latent bug the first time a slice block lands upstream. Iterate with `at<T>()` or handle strides explicitly.
- **Mind the widths at the edges.** The dtype ladder matches NumPy exactly (`CF32` is `complex64`), so keeping tensors, attributes, and environment values at consistent widths avoids conversions everywhere. The reasoning is laid out in [Type Conversion](/docs/python-block#type-conversion).
