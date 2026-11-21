---
title: Frequent Questions
description: Commonly asked questions about CyberEther, its design choices, and future plans.
order: 900
category: Resources
---

Curious why CyberEther is built the way it is? Here are answers to the questions people ask most often. If yours is not listed, feel free to open an issue on GitHub.

## Is CyberEther web-based?

No. CyberEther is a native application designed to run directly on modern hardware without Electron, Qt, or a browser wrapper. Its rendering engine talks straight to Vulkan, Metal, and WebGPU.

The edition that runs in your browser is that exact same C++ codebase compiled to WebAssembly and rendered with WebGPU. It is not a traditional web app, and none of its core engine is written in JavaScript.

## Why is CyberEther written in C++?

In one word: compatibility. At a deeper level, CyberEther is built to stay as close to the hardware as possible. Writing it in C++ gives complete control over memory layout, threading, and platform-specific optimization without being constrained by a framework. That same design lets CyberEther run smoothly inside a browser tab while scaling up to multi-GPU workstations and supercomputers.

Many newer ecosystems still sit behind layers of wrappers and abstractions that get in the way of low-level debugging and tuning. That landscape is improving as first-party support matures, though C++ remains the most practical foundation today. As John Carmack famously observed, there are "[...] externalities that can overwhelm the benefits of a language [...]". The author is also a big fan of Rust and looks forward to using it on the right project down the road.

## Why are buffer sizes stricter than in other SDR software?

Every block in CyberEther works on tensors of a fixed shape, and that shape propagates downstream from the source. If a source produces 8 batches of 4096 samples, it always emits a `[8, 4096]` tensor, and every downstream link keeps its shape from cycle to cycle. Changing a size means adjusting the source or placing a reshaping block along the chain.

Most other SDR frameworks take the opposite approach. They let each block read and emit arbitrary numbers of items while the scheduler dynamically buffers between them. That flexibility is convenient, though the runtime only discovers how much work each stage has once the pipeline is already running.

CyberEther chooses predictability instead. Knowing every tensor shape when the graph is constructed lets the engine allocate buffers once, plan kernel launches up front, and settle static branches before live samples arrive. That predictability is what makes heterogeneous pipelines across CUDA and Vulkan fast and reliable, since stray allocations or unexpected host round trips destroy throughput in the hot path.

Variable-rate workflows are still well supported through dedicated patterns:

- **Skip when not ready.** A block that has nothing to emit on a given cycle, such as a demodulator still gathering symbols, returns `SKIP`. Downstream blocks drop that frame and wait for the next one.
- **Buffer at the edge.** Ingest sources like SDR and WebSocket receive data into an internal ring buffer and emit uniform batches, isolating the graph from irregular device timing.
- **Reshape explicitly.** Blocks such as Reshape, Pad, Fold, and Decimator change tensor geometry cleanly within a single cycle.
- **Carry state across batches.** The Overlap-Add block preserves history across batch boundaries for continuous streaming convolution.

## Why does CyberEther use Jetstream as its namespace?

Originally, CyberEther was planned as the desktop application layer built on top of a separate signal-processing library called Jetstream. As the project matured, the two fused into one unified codebase. The naming convention stayed: Jetstream is the core library namespace, and CyberEther is the application.

## What is the best way to contribute to the project?

Search the codebase for `TODO` comments to find concrete tasks waiting to be tackled. Implementing new blocks and compute modules is another high-impact way to contribute. Check out the [Contributing guide](/docs/contributing) to get set up.

## What are the future plans for CyberEther?

Near-term priorities include expanding unit test coverage, improving documentation, and adding new blocks and compute modules. Long term, the goal is to push CyberEther well beyond radio communications into a fast, general-purpose accelerator for compute-intensive pipelines across many fields.

## What are the minimum requirements to run CyberEther?

There are no minimum requirements, only minimum expectations.

## Where are the easter eggs?

There are no easter eggs in CyberEther. I promise.
