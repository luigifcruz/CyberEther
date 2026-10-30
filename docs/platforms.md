---
title: Platforms
description: The three ways to run CyberEther, how they differ, and how to pick one.
order: 39
category: Usage
---

The same CyberEther runs in three shapes: as a native application on your machine, as a web application inside a browser tab, and as a remote instance that streams its interface to a browser from somewhere else. The flowgraphs are identical across all three. What changes is where the work happens, which hardware is in reach, and how much setup it takes.

## How It Works

Every platform runs the same core. The flowgraph, the scheduler, the blocks, and the interface are shared code, so a flowgraph saved on one platform opens unchanged on another. Each platform wraps that core in a different shell that decides four things:

| | Native | Web | Remote |
|---|---|---|---|
| Where compute runs | Your machine | Your browser tab | The host machine |
| How the interface is drawn | Metal or Vulkan on your GPU | WebGPU in the browser | On the host, streamed as video to your browser |
| How input arrives | Directly from the window | From the browser tab | From your browser, sent back to the host |
| Where files live | Your disk | Browser storage | The host's disk |

Native and Web both run on the device in front of you. Remote is the native application running somewhere else, with only its picture and your input traveling over the network.

## Native

The native application is the primary way to run CyberEther. It has direct access to the GPU, every compute backend the platform provides, and any hardware attached to the machine. Use it for daily work, development, and unattended deployments.

See [Native](/docs/native) for the command line, settings, and troubleshooting.

## Web

The web application is the same CyberEther compiled to WebAssembly, running entirely inside a browser tab with nothing to install. Compute runs on the CPU, and hardware access is limited to what the browser exposes. Use it to try CyberEther, give a demo, or share a flowgraph.

See [Web](/docs/web) for requirements, storage, and self-hosting.

## Remote

A remote instance is the native application, usually headless on a server, that streams its interface to a browser on another device and takes input back. The flowgraph runs at the full capacity of the host, with all of its backends and hardware. Use it to drive a headless server, a radio in another room, or a powerful machine from a laptop or a phone.

See [Remote Streaming](/docs/remote) for setup, network requirements, security, and SSH tunneling.

## Picking One

| | Native | Web | Remote |
|---|---|---|---|
| Install required | Yes | No | On the host only |
| Setup effort | Install and run | Open a link | Install on the host, share a link |
| Compute | Local, all backends | CPU via WebAssembly | Host, all native backends |
| Hardware access | Full | Browser APIs only | Full, on the host |
| Python blocks | Yes | Yes, in the browser | Yes, on the host |
| Interface latency | Lowest | Low | Depends on the network |
| Best for | Daily use, development, deployment | Trying it, demos, sharing | Headless and high-performance systems |

The rule of thumb: develop and deploy native, share via web, and use Remote Streaming to work with larger or unattended systems from anywhere on your network.
