---
title: Hardware And Devices
description: Connect software-defined radios, configure wide-band sweeps, and play audio through system sound devices.
order: 221
category: Usage
parent: blocks
---

CyberEther connects directly to software-defined radios and system audio playback devices through three built-in IO blocks: **SDR** (`sdr`), **SDR Sweep** (`sdr_sweep`), and **Audio** (`audio`). Most popular USB radios work out of the box with no extra driver packages required.

## Supported Radio Drivers

CyberEther compiles eight hardware drivers directly into the binary and detects **SoapySDR** automatically at runtime for everything else:

| Driver | Native Desktop | Browser (WebUSB) | Notes |
|---|---|---|---|
| **RTL-SDR** | Yes | Yes | Built-in driver, Bias-T supported. |
| **Airspy** | Yes | Yes | Built-in driver, Bias-T supported. |
| **Airspy HF+** | Yes | Yes | Built-in driver. |
| **HackRF** | Yes | Yes | Built-in driver, Bias-T supported, firmware sweeps require USB API `0x0104` or newer. |
| **LimeSDR** | Yes | Yes | Built-in LimeSuiteNG driver. Dual-channel requires frequency at or above 30 MHz on LimeSDR-USB. LimeSDR XTRX is supported on Linux with the `limepcie` kernel driver loaded. |
| **bladeRF** | Yes | Yes | Built-in driver, Bias-T on bladeRF 2, dual-channel timestamps require FPGA `v0.15.3` or newer. |
| **PlutoSDR** | Yes | No | Built-in driver (desktop only). |
| **BirriaSDR** | Linux and macOS | No | Built-in driver, USB 3 port required. Unbind the Wi-Fi driver and replug the dongle. |
| **SoapySDR** | Yes | No | Runtime detection of SoapySDR 0.8 or 0.9 and installed device modules. Radios with a built-in driver appear only through their built-in driver. |

### Platform Setup Notes

- **Windows:** Radios using built-in USB drivers require the **WinUSB** driver (for example via Zadig).
- **Linux:** Ensure your user has udev permissions for your USB radio so CyberEther can open the device without root privileges.
- **Browser (WebUSB):** Open the **Web Edition** in Chrome or another WebUSB-capable browser, grant the page access to the radio when prompted, and select it in the **Device** list. PlutoSDR, BirriaSDR, and SoapySDR are not available in the browser.
- **iOS and Android:** Direct USB SDR hardware is not available. Use [Remote Streaming](/docs/remote) from a desktop or embedded host instead.

## Live Streaming With the SDR Block

Use the **SDR** block (`sdr`) when you want continuous, full-time-resolution complex I/Q samples across a single tuned band.

- **Device selection:** Pick a connected radio from the **Device** dropdown. If you plug in a radio after placing the block, right-click the block and choose **Reload Block** to refresh the device list.
- **Live tuning:** Changing **Frequency** or **Gain** updates the radio immediately without restarting the stream. Changing **Sample Rate** or **Channels** reapplies the full device configuration.
- **Multichannel receivers:** On multi-receiver radios such as bladeRF 2 or LimeSDR, set **Channels** above 1 to receive multiple coherent streams in one tensor. When **Automatic Gain** is off and **Channels** is above 1, **Manual Gains** accepts one gain value per receiver. Use a **Slice** block downstream to pick out a single channel. On bladeRF 2, keep `Sample Rate * Channels` at or below 80 MS/s to stay within USB bandwidth.
- **Bias-T:** Available on RTL-SDR, Airspy, HackRF, bladeRF 2, and SoapySDR modules that expose a `biastee` setting.
- **SoapySDR stream arguments:** When using a SoapySDR device, pass comma-separated `key=value` pairs (such as `bufflen=8192`) in **Driver Arguments** under Advanced.

### Monitoring Stream Health

Every **SDR** block displays three live readouts:

- **Buffer Fill:** Smoothed fill level of the internal ring buffer from 0% to 100%. Values near 100% mean the downstream chain is falling behind.
- **Buffer Loss:** Percentage of samples overwritten before the pipeline could read them, plus any hardware `GAP` events reported by the radio. Lower the sample rate, simplify the chain, or raise **Buffer Multiplier** to absorb brief stalls.
- **Throughput:** Rate at which the chain reads data in MB/s alongside the rate the radio delivers across all channels.

## Wide-Band Sweeps With SDR Sweep

Use the **SDR Sweep** block (`sdr_sweep`) when you want to monitor a frequency span wider than your radio's instantaneous sample rate. The block hops across the band, applies a Blackman window and FFT to each hop, blends overlapping hops in power, and converts the stitched spectrum back into synthetic complex time-domain samples for downstream blocks like **Spectrum Analyzer**.

- **HackRF firmware sweeps:** On a HackRF with USB API `0x0104` or newer, CyberEther uses fast hardware-timed firmware sweeps with non-overlapping hops.
- **Dwell and Settling Time:** Raise **Dwell** to average multiple captures per hop for a calmer noise floor. If hop edges show artifacts from neighboring frequencies, raise **Settling Time** by a few milliseconds.
- **BirriaSDR and SoapySDR sweeps:** Each BirriaSDR hop takes up to about 0.75 seconds. SoapySDR sweeps are slower and depend on the underlying device module.
- **Sweep readouts:** Watch **Sweep Speed** (coverage rate in GHz/s and complete sweeps per second in Hz) alongside **Buffer Fill** and **Buffer Loss**.

## Audio Playback

The **Audio** block (`audio`) is a **playback** sink that sends mono or stereo `F32` samples to a system sound output device. CyberEther does not provide an audio microphone capture block.

Keep three rules in mind when wiring **Audio** at the end of a demodulator chain (such as **FM Demodulator** or **AM Demodulator**):

1. **Match the upstream sample rate:** The **Sample Rate** field is not read automatically from the input tensor, so set it to match the block feeding **Audio**. If audio sounds too slow and deep, or too fast and high-pitched, adjust **Sample Rate**.
2. **Automatic 48 kHz resampling:** CyberEther resamples incoming audio to 48 kHz for playback. Setting an upstream rate of 48 kHz plays directly without resampling.
3. **Pace non-realtime sources:** Hardware radios deliver samples in real time. When playing from a **File Reader** or **Signal Generator**, insert a **Throttle** block upstream so audio does not stutter or overrun the ring buffer.

To play stereo audio, provide a `channelAxis` of size 2 (left then right). If an upstream multi-head **Filter** outputs more than two channels, select one channel with a **Slice** block before **Audio**.
