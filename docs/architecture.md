---
title: Architecture
description: How the layers of CyberEther fit together, from the window to the compute kernels.
order: 78
category: Development
---

Ever wondered what actually happens between dropping a block on the canvas and samples streaming across your screen? This page is the tour. It walks the layers from the window down to the compute kernels, shows the two threads everything runs on, and traces the paths that matter, so that when you open the source, or the rest of the development documentation, you already know where you are. Twenty minutes here saves hours of spelunking later.

## The Layers

The structure is a chain of ownership. One instance owns the flowgraphs, each flowgraph owns its blocks and a scheduler, and the scheduler drives the modules through runtime implementations. Around that chain sit the shared foundations, from the render toolkit down to the tensor memory layer, and on top of it sits the compositor that draws the application. Each layer is described below, top down.

**Instance** is the application root. It owns the render window, the compositor, the optional remote endpoint, and a set of named flowgraphs. Its `compute()` and `present()` methods are the two entry points the run loops call, and each fans out to every flowgraph.

**Flowgraph** owns the graph itself: the blocks, the edges between them, the creation order, and the three shared facilities documented in [Metadata](/docs/metadata), namely the environment, the per-block metrics view, and the block metadata. All graph mutations go through it (`blockCreate`, `blockConnect`, `blockRecreate`, and friends), which is where downstream propagation, incomplete-block retries, and YAML import and export live. Each flowgraph owns one scheduler.

**Scheduler** turns the module set into an execution plan. It keeps a topological order, identifies source modules, partitions the order into segments by runtime, and coordinates the compute and present threads. The synchronous scheduler guarantees that present never blocks waiting for compute, that compute yields to present at segment boundaries, and that structural changes safely halt both threads.

**Runtimes** execute segments. The native runtime calls module hooks directly, the Python runtime routes them through the embedded interpreter bridge. A runtime owns the modules assigned to its segment and reports per-module skip and failure sets back to the scheduler.

**Blocks and Modules** are the subject of [their own page](/docs/blocks-and-modules). In one sentence, blocks are the user-facing nodes a flowgraph stores, and modules are the device-specific kernels the scheduler runs.

**Backend** provides the device contexts the modules compute with, one per device type under `src/backend/devices` (cpu, cuda, metal, vulkan, webgpu). **Render** provides the window, surfaces, and the Sakura UI toolkit, with device implementations for metal, vulkan, and webgpu. **Viewport** abstracts platform windowing and input (glfw, headless, ios, plus a capture viewport for streaming). The three are deliberately separate: a headless deployment keeps backends without a real window, and the render device need not match the compute device.

**Memory** is the tensor layer: typed n-dimensional buffers with shapes, strides, device residency, and attached attributes. Tensors flow between modules by reference through link objects, so connecting blocks never copies samples.

**Parser and Registry** are the foundations everything serializes and dispatches through. The parser owns the value type system used by configurations, the environment, and YAML files. The registry collects the static block and module registrations from the binary and from loaded plugins, and answers the lookups described in [Registration And Dispatch](/docs/blocks-and-modules#registration-and-dispatch).

**Compositor** is the application UI built on Sakura: the flowgraph editor, the pickers and menus, the Environment window, and the themes. It talks to the rest of the system exclusively through the flowgraph and view APIs, which keeps the door open for alternative frontends and for the headless mode where no compositor runs.

## The Two Loops

CyberEther runs on two long-lived threads started by the run target:

- The **compute thread** loops `instance->compute()`, which runs each flowgraph's scheduler cycle back to back. Its natural rate comes from source modules blocking in `hasPendingCompute()` until data is available. A graph with no blocking source is free-running.
- The **present thread** loops `instance->present()`, which begins a render frame, lets each flowgraph present its surfaces, draws the compositor, and ends the frame. It runs at display rate.

The synchronous scheduler mediates between them with a priority rule: present never waits for compute, and compute checks for a pending present request at every segment boundary and yields. This keeps the UI responsive regardless of how heavy the DSP is.

Graph mutations can originate from either thread, and from others besides, since the compositor mutates on the present thread, host code and remote endpoints mutate from their own threads, and the incomplete-block retry mutates from the compute thread. The flowgraph serializes all of it behind a mutation lock, and the scheduler halts compute around structural changes, so block authors never see a half-mutated graph.

Two consequences worth internalizing:

- Anything called from the compute path must not block, or the entire flowgraph stalls. Sources are the single exception, and only inside `hasPendingCompute()`.
- Render resources are bound through queues processed at frame boundaries, so creating blocks with surfaces is safe from any thread. That is what makes the retry and remote paths sound.

## Paths Worth Tracing

**Creating a block.** `Flowgraph::blockCreate` builds the block through the registry, resolves the requested input links against existing block outputs, and runs the block lifecycle. The block's `create()` instantiates modules, each module registers with the scheduler, and the scheduler rebuilds its topological order and segments. A block whose inputs cannot resolve yet is kept in the incomplete state rather than rejected.

**A compute cycle.** The scheduler snapshots the source modules, polls each source's `hasPendingCompute()` until all report ready (this is where a blocking source paces the graph), then executes the segments in order. Each runtime runs its modules' `computeSubmit`, honoring the skip propagation and failure semantics described in [the compute contract](/docs/blocks-and-modules#compute-contract). Module failures are collected and converted into errored blocks after the cycle, and the rest of the graph keeps running.

**Editing a configuration.** The compositor writes the change through `blockReconfigure`, which synchronizes with the scheduler and calls the block's `reconfigure`. An in-place change ends there. A `RECREATE` answer makes the flowgraph capture the state of the block and everything downstream, destroy them in reverse order, and recreate them in forward order with connections intact.

**Loading a flowgraph file.** The YAML is parsed into a document, migrated across format versions if needed, topologically sorted so producers exist before consumers, and replayed through the same `blockCreate` path the editor uses. There is exactly one way blocks come into existence, whatever the origin.

## Where To Go Next

To build something inside a flowgraph, read [Blocks And Modules](/docs/blocks-and-modules) and [Metadata](/docs/metadata). To ship it separately, read [Creating Plugins](/docs/plugins). To script instead of compile, read [Python Block](/docs/python-block).
