---
title: Blocks
description: The catalog of blocks that ship with CyberEther.
order: 41
category: Usage
---

This is the catalog of every block that ships with CyberEther, grouped by the domain they appear under in the block picker. The type string in the second column is the block's stable identifier: it is what flowgraph files store and what programmatic creation uses. Each block also carries a longer description with usage notes, visible in the documentation pane of the editor, so this page lists the one-line summaries and leaves the details to the application itself.

Plugins can extend this catalog with their own blocks. Installing one is covered in [Installing Plugins](/docs/installing-plugins), and writing one is covered in [Blocks And Modules](/docs/blocks-and-modules). For custom processing without C++, see the [Python block](/docs/python-block).

## Core

Tensor manipulation and general plumbing: shaping, casting, arithmetic, and the utility blocks that glue chains together.

| Block | Type | Summary |
|---|---|---|
| Add | `add` | Element-wise addition. |
| Arithmetic | `arithmetic` | Reduces a tensor along an axis using an arithmetic operation. |
| Cast | `cast` | Casts the input to a type. |
| Comparator | `comparator` | Compares inputs for numerical similarity. |
| Duplicate | `duplicate` | Copies and transfers signal data. |
| Expand Dims | `expand_dims` | Inserts a new dimension of size 1 at a specified axis. |
| Flatten | `flatten` | Flattens a tensor to one dimension. |
| Invert | `invert` | Alternating sign inversion for FFT shift. |
| Multiply | `multiply` | Element-wise multiplication. |
| Multiply Constant | `multiply_constant` | Multiplies input by a constant value. |
| Ones Tensor | `ones_tensor` | Creates a tensor filled with ones. |
| Pad | `pad` | Adds zeros to the end of a tensor. |
| Permutation | `permutation` | Reorders tensor axes with a user-defined permutation. |
| Python | `python` | Runs custom Python compute code. |
| Range | `range` | Scales input to a specified range. |
| Reshape | `reshape` | Changes the shape of a tensor. |
| Slice | `slice` | Extracts a subset of a tensor. |
| Squeeze Dims | `squeeze_dims` | Removes a dimension of size 1 at a specified axis. |
| Throttle | `throttle` | Limits data flow rate by introducing time delays. |
| Unpad | `unpad` | Removes padding from a tensor. |

## DSP

The signal processing chain: transforms, filters, demodulators, and generators.

| Block | Type | Summary |
|---|---|---|
| ADS-B Decoder | `adsb` | Decodes ADS-B Mode S frames and maps aircraft positions. |
| AGC | `agc` | Automatic Gain Control. |
| AM Demodulator | `am` | Demodulates an amplitude modulated signal. |
| Amplitude | `amplitude` | Calculates the amplitude of a signal in decibels. |
| Decimator | `decimator` | Decimates a signal by summing along an axis. |
| FFT | `fft` | Performs the Fast Fourier Transform. |
| FM Demodulator | `fm` | Demodulates a frequency modulated signal. |
| Filter | `filter` | Filters input signal with a FIR bandpass filter. |
| Filter Engine | `filter_engine` | Filters a signal using FIR filter coefficients. |
| Filter Taps | `filter_taps` | Generates FIR bandpass filter coefficients. |
| Fold | `fold` | Folds the input signal along a specified axis. |
| Overlap Add | `overlap_add` | Sums overlap with buffer for streaming convolution. |
| PSK Demodulator | `psk_demod` | Demodulates PSK signals with carrier and timing recovery. |
| RRC Filter | `rrc_filter` | Root raised cosine matched filter for PSK modulation. |
| Signal Generator | `signal_generator` | Generates synthetic waveforms, noise, and chirps. |
| Spectrum Engine | `spectrum_engine` | Computes spectra with windowing, FFT, and optional scaling. |
| Squelch | `squelch` | Passes input only when signal strength is above a threshold. |
| Window | `window` | Generates a Blackman window function. |

## IO

Sources and sinks: hardware, files, audio, and the network.

| Block | Type | Summary |
|---|---|---|
| Audio | `audio` | Audio playback device interface. |
| File Reader | `file_reader` | Reads raw binary signal data from a file. |
| File Writer | `file_writer` | Writes raw binary signal data to a file. |
| Soapy SDR | `soapy` | Interface for SoapySDR devices. |
| WebSocket | `websocket` | Receives data streams over WebSocket. |

## Visualization

Blocks that render a surface into their node.

| Block | Type | Summary |
|---|---|---|
| Constellation | `constellation` | Displays a constellation scatter plot. |
| Frame | `frame` | Displays a frame buffer on a surface. |
| Lineplot | `lineplot` | Displays data in a line plot visualization. |
| Note | `note` | Displays formatted markdown text inside a node. |
| Spectrogram | `spectrogram` | Displays a spectrogram of data. |
| Waterfall | `waterfall` | Shows frequency spectrum over time as a scrolling waterfall. |
