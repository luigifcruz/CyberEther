---
title: Frequent Questions
description: Commonly asked questions about CyberEther, its design choices, and future plans.
order: 5
category: Getting Started
---

This page answers some of the most common questions about CyberEther. If you have a question that is not answered here, feel free to open an issue on GitHub.

## Is CyberEther web-based?

No. CyberEther is a native application that can run on any modern device. It does not use a Browser, Electron, Qt, or any other multi-platform framework. It is built on top of low-level APIs such as Vulkan, Metal, and WebGPU. The CyberEther version that runs in the browser is a fully-featured application that is compiled to WebAssembly and runs on top of WebGPU. It is not a web-based application in the traditional sense and it does not use JavaScript for the core functionality.

## Why is CyberEther written in C++?

The short answer to this question is that compatibility is king. The longer answer is that one of the design choices is to be as low-level as possible. This allows me to have full control over the code and to be able to optimize it for a specific platform without being locked down by a framework. That is why CyberEther can run inside the browser while being able to scale towards a supercomputer. Currently, one of the problems with other languages is that they have too many wrappers and abstractions that make it difficult to debug and optimize the code. I expect this to change in the future as first-party support grows but for now, C++ is the best option. As John Carmack once said, "[...] externalities that can overwhelm the benefits of a language [...]". I am also a big fan of Rust and I am looking forward to finding a good project to use it in the future.

## What is the best way to contribute to the project?

The code is riddled with TODO comments. These are a good place to start. Another way to contribute is by implementing new blocks and modules.

## Why CyberEther uses Jetstream as the namespace?

At the beginning of the project, CyberEther was meant to be only an application that utilized Jetstream as a library. But as the project evolved, the two things became one. In the end, I decided to keep Jetstream as the name of the library and CyberEther as the name of the application.

## How CyberEther compares to GNU Radio?

First of all, GNU Radio is an amazing project. It is a very powerful tool and it is used by many people and institutions around the world. Developing CyberEther as a separate project without the burden of backward compatibility and legacy code allowed me to explore new ideas that would be time-consuming to implement in GNU Radio. I think of CyberEther as a playground for new ideas that can be later integrated into GNU Radio if they prove to be useful. Beyond radio, CyberEther has the potential to be used in other domains such as machine learning, computer vision, and robotics. The goal is to make CyberEther a general-purpose acceleration tool for compute-intensive pipelines.

## Why are buffer sizes stricter than in other SDR software?

Every block in CyberEther runs on tensors of a fixed shape, and that shape flows downstream from the block that produces the data. A source configured for 8 batches of 4096 samples always emits a tensor of shape `[8, 4096]`, and every connection in the graph keeps its shape from one cycle to the next. If a block needs a different size, the size has to change at the source or through a block that reshapes the stream along the way.

Most SDR software works the other way around, letting each block consume and produce however many items it wants and leaving it to the scheduler to buffer between them. That flexibility is convenient, but it also means the runtime only learns how much data each stage handles, and when a stage has enough to run, while the pipeline is already running. CyberEther takes the opposite trade. Because every tensor shape is known when the graph is built, the graph can prepare its buffers once, plan its kernel launches, and settle the pipeline before it runs. That predictability is what makes it practical to run the same flowgraph across heterogeneous devices such as CUDA and Vulkan, where a stray allocation or a host round trip in the hot path is exactly what a fast pipeline cannot afford.

The practical uses of variable-size streaming are still covered, just in a different way. A block that has nothing to emit on a given cycle, such as a demodulator that has not accumulated enough symbols yet, returns `SKIP` and every block downstream simply waits for the next cycle. Ingest sources such as SoapySDR and WebSocket receive into an internal ring buffer and emit fixed-size batches from it, so the device delivery pattern never reaches the graph. Blocks such as Reshape, Pad, Fold, and Decimator change the shape along the way, each within one cycle of data, and Overlap-Add carries state across batch boundaries for filtering that would otherwise need a continuous stream.

## What are the future plans for CyberEther?

The plans for CyberEther include adding unit tests, improving documentation, and implementing new blocks and modules. The goal is to continue expanding the capabilities of CyberEther beyond radio communications and make it a powerful tool for accelerating compute-intensive pipelines in various domains.

## What are the minimum requirements to run CyberEther?

There are no minimum requirements, only minimum expectations.

## Where are the easter eggs?

There are no easter eggs in CyberEther. I promise.
