---
title: Blocks
description: The catalog of blocks that ship with CyberEther.
order: 220
category: Usage
---

Every node you place on a CyberEther canvas is a block. This catalog lists every built-in block, organized by the five domains shown in the block picker: **Core**, **DSP**, **IO**, **Visualization**, and **ML**.

The **Type** column lists each block's stable identifier, which is what gets written into `.yml` flowgraph files and used when creating blocks programmatically. Every block also ships with detailed usage notes and parameter descriptions inside the app itself: right-click any block on the canvas and select **Documentation** to read it.

## Common Pipeline Patterns

If you are browsing the catalog to figure out which blocks work together, here are three classic chains:

- **Spectrum analysis:** Connect `signal_generator` (set to `CF32`) or `sdr` to `spectrum_engine` (with **Enable Scale** turned on), then to `lineplot` and `waterfall`, or feed `spectrum_analyzer` directly.
- **Live FM radio:** Connect `sdr` to `filter`, `fm`, `rational_resampler`, and `audio`.
- **Custom prototyping:** Connect `sdr` or `file_reader` to `python`, then to `constellation` or `lineplot`.

## Core

Tensor plumbing, math, reshaping, metadata helpers, and custom Python scripting that glue signal chains together.

| Block | Type | Summary |
|---|---|---|
| Add | `add` | Adds two tensors element by element, with broadcasting. |
| Arithmetic | `arithmetic` | Reduces a tensor along an axis using an arithmetic operation. |
| Cast | `cast` | Converts input elements to a target data type. |
| Comparator | `comparator` | Compares inputs for numerical similarity. |
| Complex Merge | `complex_merge` | Combines two real signals into one complex signal. |
| Complex Split | `complex_split` | Splits a complex signal into real and imaginary components. |
| Duplicate | `duplicate` | Copies and transfers signal data across buffers or devices. |
| Expand Dims | `expand_dims` | Inserts a new dimension of size 1 at a chosen axis. |
| Flatten | `flatten` | Collapses a tensor into one dimension. |
| Multiply | `multiply` | Multiplies two tensors element by element, with broadcasting. |
| Multiply Constant | `multiply_constant` | Scales input elements by a constant value. |
| Ones Tensor | `ones_tensor` | Produces a constant tensor filled with ones. |
| Pad | `pad` | Appends zeros to the end of a tensor. |
| Permutation | `permutation` | Reorders tensor axes using a custom permutation. |
| Python | `python` | Executes custom Python compute code on tensors and metadata. |
| Range | `range` | Normalizes input values using an affine range mapping. |
| Remove Indices | `remove_indices` | Removes selected entries by index. |
| Reshape | `reshape` | Changes the shape of a tensor without copying samples. |
| Signal Attributes | `signal_attributes` | Attaches or overrides signal attributes on a tensor. |
| Signal Axes | `signal_axes` | Assigns batch, channel, and sample roles to tensor dimensions. |
| Slice | `slice` | Extracts a sub-range of a tensor as a view. |
| Squeeze Dims | `squeeze_dims` | Removes a dimension of size 1 at a chosen axis. |
| Throttle | `throttle` | Limits pipeline flow rate by pacing execution intervals. |
| Unpad | `unpad` | Strips padding from the end of a tensor. |

## DSP

Signal-processing transforms, filters, demodulators, resamplers, and waveform generators.

| Block | Type | Summary |
|---|---|---|
| ADS-B Decoder | `adsb` | Decodes ADS-B Mode S frames and maps live aircraft positions. |
| AGC | `agc` | Normalizes signal amplitude to a target RMS level. |
| AM Demodulator | `am` | Demodulates an amplitude-modulated signal. |
| Amplitude | `amplitude` | Computes signal magnitude in decibels. |
| Decimator | `decimator` | Downsamples a signal along its sample axis. |
| FFT | `fft` | Computes the Fast Fourier Transform. |
| FM Demodulator | `fm` | Demodulates a frequency-modulated signal. |
| Filter | `filter` | Applies an FIR bandpass filter to the input signal. |
| Filter Engine | `filter_engine` | Convolves a signal with supplied FIR filter coefficients. |
| Filter Taps | `filter_taps` | Generates FIR bandpass filter coefficients. |
| Fold | `fold` | Folds the input signal along its sample axis. |
| Invert | `invert` | Shifts a signal so the following FFT is centered around zero frequency. |
| Overlap Add | `overlap_add` | Accumulates overlap buffers for continuous streaming convolution. |
| Phase Correction | `phase_correction` | Applies a phase rotation to a complex signal. |
| PSK Demodulator | `psk_demod` | Demodulates PSK signals with carrier and timing recovery. |
| Rational Resampler | `rational_resampler` | Resamples a signal by a rational factor. |
| RRC Filter | `rrc_filter` | Applies a root-raised-cosine matched filter to a PSK signal. |
| Signal Generator | `signal_generator` | Produces synthetic waveforms, noise, and frequency chirps. |
| Spectrum Engine | `spectrum_engine` | Computes spectra with windowing, FFT, and optional scaling. |
| Squelch | `squelch` | Passes samples through only when signal strength exceeds a threshold. |
| Window | `window` | Generates a Blackman window function. |

## IO

Sources and sinks connecting CyberEther to radios, audio playback devices, files, and network streams. For radio drivers, WebUSB, sweeps, and audio setup, see [Hardware And Devices](/docs/hardware).

| Block | Type | Summary |
|---|---|---|
| Audio | `audio` | Plays mono or stereo audio streams through a system sound output device (playback only). |
| File Reader | `file_reader` | Reads SigMF recordings or raw binary signal samples from a file on disk. |
| File Writer | `file_writer` | Writes SigMF recordings or raw binary signal samples to a file on disk. |
| SDR | `sdr` | Receives live samples from a connected software-defined radio. |
| SDR Sweep | `sdr_sweep` | Sweeps a radio across a wide frequency span and stitches the band together. |
| TCP Receiver | `tcp_receiver` | Receives streaming tensor data over TCP. |
| TCP Transmitter | `tcp_transmitter` | Transmits streaming tensor data over TCP. |
| UDP Receiver | `udp_receiver` | Receives streaming tensor data over UDP. |
| UDP Transmitter | `udp_transmitter` | Transmits streaming tensor data over UDP. |
| WebSocket Receiver | `websocket_receiver` | Receives streaming tensor data over WebSocket. |
| WebSocket Transmitter | `websocket_transmitter` | Transmits streaming tensor data over WebSocket. |
| ZeroMQ Receiver | `zeromq_receiver` | Receives streaming tensor data over ZeroMQ. |
| ZeroMQ Transmitter | `zeromq_transmitter` | Transmits streaming tensor data over ZeroMQ. |

## Visualization

Interactive views and live documentation notes rendered on the GPU directly inside their node or in detached dock tiles.

| Block | Type | Summary |
|---|---|---|
| Constellation | `constellation` | Renders an I/Q constellation scatter plot. |
| Eye Diagram | `eye_diagram` | Overlays an oversampled signal two symbols at a time to inspect timing and signal quality. |
| Frame | `frame` | Displays 2D images and heat maps. |
| Lineplot | `lineplot` | Draws real-time 1D line plots. |
| Note | `note` | Renders formatted Markdown text and live environment stats inside a node. |
| Spectrum Analyzer | `spectrum_analyzer` | Combines a live spectrum trace and scrolling waterfall for complex samples. |
| Waterfall | `waterfall` | Plots frequency spectrum history over time as a scrolling waterfall. |
| 3D Waterfall | `waterfall_3d` | Renders spectrum history as an interactive 3D surface. |

## ML

Neural-network inference on live signal tensors, available in builds compiled with ONNX Runtime (`-Dinference=enabled`).

| Block | Type | Summary |
|---|---|---|
| ONNX Inference | `onnx_inference` | Runs an ONNX model directly on incoming tensors. |

## Extending the Catalog

Need a block that is not listed here? You have three options:

- Write custom processing in minutes with no compiler using the [Python Block](/docs/python-block).
- Add community or third-party `.cep` bundles by following [Installing Plugins](/docs/installing-plugins).
- Build native high-performance C++ blocks and compute modules with [Blocks And Modules](/docs/blocks-and-modules).
