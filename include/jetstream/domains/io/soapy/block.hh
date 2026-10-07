#ifndef JETSTREAM_DOMAINS_IO_SOAPY_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Soapy : public Block::Config {
    std::string modulePath = "";
    std::string deviceString = "";
    std::string streamString = "";
    std::string antenna = "";
    F32 frequency = 96.9e6;
    F32 frequencyStep = 1000000.0;
    F32 sampleRate = 2.0e6;
    bool automaticGain = true;
    bool biasTee = false;
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_BLOCK_TYPE(soapy);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(modulePath, deviceString, streamString,
                     antenna, frequency, frequencyStep, sampleRate,
                     automaticGain, biasTee, numberOfBatches,
                     numberOfTimeSamples, bufferMultiplier);
    JST_BLOCK_DESCRIPTION(
        "Soapy SDR",
        "Receives samples from a SoapySDR radio.",
        "# Soapy SDR\n"
        "\n"
        "Streams complex samples from a software-defined radio through SoapySDR. It "
        "usually starts a chain, feeding a Spectrum Analyzer to watch the band or a "
        "Filter that isolates one station.\n"
        "\n"
        "- **The radio paces the chain.** Each buffer comes out only once the radio has"
        " delivered all of its samples.\n"
        "- **A slow chain loses samples.** When the chain falls behind, the oldest "
        "samples are overwritten and Buffer Loss rises.\n"
        "- **Automatic Gain off keeps the last gain.** The block has no manual gain "
        "setting.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | CF32 | Batches x Samples, on a `batchAxis` then a "
        "`sampleAxis`. Sets `frequency` to Frequency and `sampleRate` to Sample Rate. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Device** | None | None and the radios found | Radio to open. A saved radio"
        " that is missing shows as Configured device. |\n"
        "| **Antenna** | Default | Default and the radio's receive ports | Receive "
        "port. Default keeps the driver's choice. |\n"
        "| **Frequency** | 96.9 MHz | Within the radio's range | Center frequency of "
        "the output. |\n"
        "| **Sample Rate** | 2 MHz | Within the radio's range | Complex samples per "
        "second, which is also the bandwidth captured. |\n"
        "| **Automatic Gain** | On | On, Off | Lets the radio set its own gain. |\n"
        "| **Bias-T** | Off | On, Off | Powers an active antenna through the cable, on "
        "radios that support it. |\n"
        "| **Batches** | 8 | 1 or more | Batches per output buffer. |\n"
        "| **Samples** | 8,192 | 1 or more | Samples per batch, and the FFT size "
        "downstream. |\n"
        "| **Buffer Multiplier** | 4 | 1 or more | Output buffers the internal buffer "
        "holds before old samples are overwritten. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Buffer Health** | How full the internal buffer is, smoothed, from 0 to "
        "100%. Near 100% means the chain is falling behind. |\n"
        "| **Buffer Loss** | Share of samples overwritten before the chain read them "
        "since the stream started, plus device overflows as OVF. |\n"
        "| **Throughput** | Rate the chain reads in MB/s, next to the rate the radio "
        "should send at 8 bytes per sample. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume a radio that supports the chosen rate.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Broadcast FM band | Defaults | 8 batches x 8,192 complex samples at 2 MS/s, "
        "30.5 buffers per second |\n"
        "| Finer spectrum | Batches 2, Samples 32,768 | 2 batches x 32,768 complex "
        "samples at 2 MS/s, 30.5 buffers per second |\n"
        "| Low latency | Batches 1, Samples 2,048 | 2,048 complex samples at 2 MS/s, "
        "one buffer every 1.02 ms |\n"
        "| Wider view | Sample Rate 10 MHz | 8 batches x 8,192 complex samples at 10 "
        "MS/s, 153 buffers per second |\n"
        "| Ride out stalls | Buffer Multiplier 16 | 8 batches x 8,192 complex samples, "
        "with 0.52 seconds held in the internal buffer |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that no device is selected | Device is None, the default |"
        " Pick a radio in Device. |\n"
        "| The block reports that no SoapySDR devices were found | The radio is "
        "unplugged, or the browser has no access to it | Plug it in and pick it again "
        "in Device. In a browser, see Setup. |\n"
        "| The block reports that it failed to open the device | Another program holds "
        "the radio, or its driver failed | Close the other program and pick the radio "
        "again. |\n"
        "| The block reports that the frequency or sample rate is not supported | The "
        "value is outside the range the radio reports | Pick a value the radio "
        "supports. While running, the log warns instead and the old value stays. |\n"
        "| Buffer Loss rises above 0% | The chain reads slower than the radio sends | "
        "Lower Sample Rate or lighten the chain. Raise Buffer Multiplier for short "
        "stalls. |\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Platform | Requirement |\n"
        "|---|---|\n"
        "| macOS, Linux, Windows | Built-in drivers for RTL-SDR, HackRF, Airspy, "
        "LimeSDR, and bladeRF. |\n"
        "| Browser | The same radios over WebUSB, once the page has been given access "
        "to the radio. |\n"
        "| iOS | Radios shared over the network by a SoapyRemote server. |\n"
        "| Android | Not available. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "A receive thread reads the radio into a ring buffer, and each cycle copies out"
        " one full buffer once enough samples have arrived. The Bias-T is switched off "
        "again when the block closes the radio.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Spectrum Analyzer** shows the band it receives.\n"
        "- **Filter** isolates one station from the capture.\n"
        "- **File Writer** records the samples to disk.\n"
        "- **Signal Generator** stands in when no radio is attached."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_BLOCK_HH
