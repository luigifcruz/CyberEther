---
title: Architecture
description: How the layers of CyberEther fit together, from the window to the compute kernels.
order: 310
category: Development
---

Ever wondered what happens between dropping a block onto the canvas and watching live samples stream across your screen? This guide walks you through CyberEther's architecture from top to bottom: the ownership hierarchy, the two threads that drive everything, and the four core execution paths. Spending a few minutes here will save you hours of navigating the codebase later.

## The Ownership Stack

CyberEther is organized as a clean chain of ownership surrounded by shared low-level foundations:

```text
  Compositor (Sakura UI: editor, pickers, themes, Environment tile)
       │ talks exclusively through Flowgraph and View APIs
       ▼
  Instance (application root, render window, optional remote endpoint)
       └── Flowgraph (blocks, edges, metadata, environment, metrics)
             └── Scheduler (topological order, segments, thread sync)
                   └── Runtimes (Native C++, Embedded Python)
                         └── Modules (device-specific compute & surface kernels)

  Shared Foundations:
  Backend (CPU/CUDA/Metal/Vulkan/WebGPU) · Render · Viewport · Memory (Tensors) · Parser & Registry
```

### Top-Down Layer Breakdown

- **Instance** is the application root. It owns the render window, the compositor, the optional remote streaming endpoint, and a collection of named flowgraphs. Its `compute()` and `present()` entry points fan out to every active flowgraph.
- **Flowgraph** owns the graph structure itself: blocks, edges, creation order, and the three metadata facilities covered in [Metadata](/docs/metadata) (environment, per-block metrics view, and block metadata). Every graph mutation flows through it (`blockCreate`, `blockConnect`, `blockRecreate`, and friends), handling downstream propagation, incomplete-block retries, and YAML import/export. Each flowgraph owns one scheduler.
- **Scheduler** compiles the active module set into an execution plan. It maintains a topological sort, detects source modules, groups modules into segments by runtime, and coordinates the compute and present threads. Branches driven by independent sources execute independently, each module consumes one frame per input per step, and multi-input joins only fire when all inputs derive from the same source frames (see [Compute Contract](/docs/module-contract#compute-contract)).
- **Runtimes** execute scheduler segments. The native runtime invokes C++ module hooks directly, while the Python runtime bridges calls into the embedded interpreter. Each runtime owns the modules in its segment and reports skipped, stalled, and failed modules back to the scheduler.
- **Blocks and Modules** form the two halves of every node. Blocks are the user-facing nodes stored in a flowgraph, while modules are the device-specific kernels executed by the scheduler. See [Blocks And Modules](/docs/blocks-and-modules) for the full design.
- **Backend, Render, and Viewport** stay deliberately decoupled:
  - **Backend** supplies compute device contexts under `src/backend/devices` (`cpu`, `cuda`, `metal`, `vulkan`, `webgpu`).
  - **Render** manages windows, surfaces, and the Sakura UI toolkit across `metal`, `vulkan`, and `webgpu`.
  - **Viewport** abstracts platform windowing and input (`glfw`, `headless`, `ios`, and a capture viewport for remote streaming).
  Keeping these three apart lets headless deployments run compute backends without a real window, and lets your compute device differ from your rendering GPU.
- **Memory** provides the tensor layer: typed multidimensional buffers with shapes, strides, device residency, and attached attributes. Tensors pass between modules by reference through link objects, so wiring blocks never copies sample data.
- **Parser and Registry** underpin serialization and discovery. The parser defines the typed value system shared by block configs, the environment, and YAML files. The registry collects static block and module registrations from both the main binary and loaded plugins (see [Registration And Dispatch](/docs/blocks-and-modules#registration-and-dispatch)).
- **Compositor** implements the visual application on top of Sakura (the flowgraph editor, block pickers, menus, Environment tile, and themes). Because it communicates strictly through flowgraph and view APIs, alternative frontends such as [Superluminal](/docs/superluminal) remain easy to build, and headless runs can pair a headless viewport with or without the default compositor (for example, keeping the compositor active when streaming over [Remote Streaming](/docs/remote)).

## The Two Execution Loops

Every CyberEther run target starts two long-lived threads:

```text
  Compute Thread                          Present Thread
  ┌───────────────────────────────┐       ┌───────────────────────────────┐
  │ loop: instance->compute()     │       │ loop: instance->present()     │
  │  • poll source readiness      │       │  • begin render frame         │
  │  • run runtime segments       │◀─yield│  • present block surfaces     │
  │  • sleep if idle (<= 100 ms)  │       │  • draw compositor UI         │
  └───────────────────────────────┘       └───────────────────────────────┘
```

- **The compute thread** loops `instance->compute()`, running one scheduler cycle across all flowgraphs back to back. Sources answer `hasPendingCompute()` immediately and wake the scheduler whenever fresh data arrives. If no flowgraph makes progress during a round, the instance sleeps until a source notifies, a throttled module comes due, or 100 ms elapses. A graph whose sources are always ready runs continuously at full speed.
- **The present thread** loops `instance->present()` at display rate, starting a render frame, presenting each flowgraph's visual surfaces, drawing the compositor, and ending the frame.

The synchronous scheduler enforces one strict priority rule: **present never waits for compute**. At every segment boundary, the compute thread checks for a pending present request and yields immediately. That keeps the interface responsive no matter how heavy the DSP pipeline gets.

Graph mutations can arrive from multiple threads: the compositor mutates on the present thread, host code and remote endpoints mutate from their own threads, and incomplete-block retries mutate from the compute thread. The flowgraph serializes every mutation behind a single lock, and the scheduler halts compute around structural edits so block authors never observe a half-built graph.

> [!IMPORTANT]
> Three architectural consequences to keep in mind when writing blocks:
> 1. **Never block on the compute path.** Blocking inside `computeSubmit()` or `hasPendingCompute()` stalls every branch of every flowgraph.
> 2. **Shapes are fixed at creation.** Buffers are allocated once when a block is created and reused every cycle. Any change that alters tensor geometry is a graph mutation, not a live reallocation.
> 3. **Surface binding is thread-safe.** Render resources bind through queues drained at frame boundaries, making it safe to create visual blocks from retry loops or remote threads.

## Four Paths Worth Tracing

### 1. Creating a Block

Calling `Flowgraph::blockCreate` instantiates the block via the registry, resolves requested input links against existing outputs, and runs the block lifecycle. Because `define()` runs before semantic validation, even an invalid block still exposes its ports and editable controls in the UI. Inside `create()`, the block builds its child modules, registers them with the scheduler, and triggers a rebuild of the topological order and segments. If required inputs are not ready yet, the block stays in the **incomplete** state rather than failing outright.

### 2. Running a Compute Cycle

At the start of a cycle, the scheduler snapshots all source modules and polls each source's `hasPendingCompute()` once without waiting. Next, it walks segments in topological order and selects every module that satisfies three conditions:

1. Every input carries an unread frame.
2. All inputs trace back to the same source frames.
3. Every downstream consumer has finished reading the module's previous output.

Each runtime executes `computeSubmit()` on its selected modules while honoring skip, stall, and failure semantics (see [Compute Contract](/docs/module-contract#compute-contract)). Any module failures are collected after the cycle and turned into errored blocks while the rest of the graph keeps running. If nothing ran, the scheduler reports its next throttle deadline so the instance can decide whether to sleep.

### 3. Editing a Configuration

When you edit a field in the editor, the compositor calls `blockReconfigure`, which synchronizes with the scheduler and invokes the block's reconfiguration flow:

- An **in-place** update finishes immediately without rebuilding downstream blocks.
- Returning `RECREATE` tears down and rebuilds the affected blocks with the new settings.
- A **semantic validation error** still rebuilds and publishes the candidate as an errored block, keeping downstream connections intact as unresolved links so you can fix the typo and recover cleanly.
- A **failure while applying an already-validated edit** rolls back the previous working graph state and returns the error.

### 4. Loading a Flowgraph File

Loading a `.yml` file parses the YAML document, migrates older schema versions when needed, topologically sorts blocks so producers come before consumers, and replays every block through `blockCreate`. Whether a block arrives from the editor, a saved file, or C++ host code, it always follows the exact same creation path.

## Where to Go Next

| What You Want to Build | Read Next |
|---|---|
| Native C++ blocks and compute modules | [Blocks And Modules](/docs/blocks-and-modules) |
| Zero-copy multi-device buffers and views | [Tensors](/docs/tensors) |
| Environment keys, tensor attributes, and metrics | [Metadata](/docs/metadata) |
| Embedded C++ and Python plotting windows | [Superluminal](/docs/superluminal) |
| Out-of-tree `.cep` plugin bundles | [Creating Plugins](/docs/plugins) |
| Quick custom processing without C++ | [Python Block](/docs/python-block) |
