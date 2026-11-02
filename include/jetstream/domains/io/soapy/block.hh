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
    F32 manualGain = 0.0f;
    bool biasTee = false;
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_BLOCK_TYPE(soapy);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(modulePath, deviceString, streamString,
                     antenna, frequency, frequencyStep, sampleRate,
                     automaticGain, manualGain, biasTee, numberOfBatches,
                     numberOfTimeSamples, bufferMultiplier);
    JST_BLOCK_DESCRIPTION(
        "SoapySDR",
        "Receives samples through an installed SoapySDR.",
        "# SoapySDR\n"
        "\n"
        "Streams complex samples from a radio through a SoapySDR installed on the "
        "computer, using its device modules. It usually starts a chain, feeding a "
        "Spectrum Analyzer to watch the band or a Filter that isolates one station.\n"
        "\n"
        "- **SoapySDR is not bundled.** The block loads SoapySDR 0.8 from the system "
        "the first time it is used.\n"
        "- **Changes drop older samples.** Retuning drops buffered samples, but the "
        "driver can still deliver a few older ones.\n"
        "- **A slow chain loses samples.** When the chain falls behind, the oldest "
        "samples are overwritten and Buffer Loss rises.\n"
        "- **Automatic Gain off uses Manual Gain.** The saved gain is reapplied after "
        "tuning and clamped to the radio's gain range.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | CF32 | Batches x Samples, on a `batchAxis` then a "
        "`sampleAxis`. Sets `frequency` and `sampleRate` to the values applied to these"
        " samples. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Device** | None | None and the radios found | Radio to open, saved as "
        "SoapySDR arguments. A saved radio that is missing shows as Configured device. "
        "|\n"
        "| **Antenna** | Default | Default and the radio's receive ports | Receive "
        "port. Default keeps the driver's choice. |\n"
        "| **Frequency** | 96.9 MHz | Within the radio's range | Center frequency of "
        "the output. |\n"
        "| **Sample Rate** | 2 MHz | Within the radio's range | Complex samples per "
        "second, which is also the bandwidth captured. |\n"
        "| **Automatic Gain** | On | On, Off | Lets the radio set its own gain. |\n"
        "| **Manual Gain** | 0 dB | Within the radio's gain range | Requested receive "
        "gain, shown when Automatic Gain is off. Limits follow the tuned frequency. |\n"
        "| **Bias-T** | Off | On, Off | Powers an active antenna through the cable, on "
        "radios that support it. Hidden once the radio is open and lacks it. |\n"
        "| **Batches** | 8 batches | 1 or more | Batches per output buffer. |\n"
        "| **Samples** | 8,192 samples | 1 or more | Samples per batch, and the FFT "
        "size downstream. |\n"
        "| **Buffer Multiplier** | 4x | 1 or more | Output buffers the internal buffer "
        "holds before old samples are overwritten. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Buffer Fill** | How full the internal buffer is, smoothed, from 0 to "
        "100%. Near 100% means the chain is falling behind. |\n"
        "| **Buffer Loss** | Share of samples overwritten before the chain read them "
        "since the stream started. Overflows the radio reported show next to it as OVF."
        " |\n"
        "| **Throughput** | Rate the chain reads in MB/s, next to the rate the radio "
        "sends at 8 bytes per sample. |\n"
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
        "| The block reports that SoapySDR 0.8 was not found | No compatible SoapySDR "
        "library is installed, or this is a browser or mobile build | Install SoapySDR "
        "and its device modules, or use the SDR block. |\n"
        "| The block reports that no device is selected | Device is None, the default |"
        " Pick a radio in Device. |\n"
        "| The block reports that no SoapySDR devices were found | The radio is "
        "unplugged, or no installed module supports it | Plug it in and install the "
        "SoapySDR module for the radio. |\n"
        "| The block reports that it failed to open the device | Another program holds "
        "the radio, or its driver failed | Close the other program and pick the radio "
        "again. |\n"
        "| Buffer Loss rises above 0% | The chain reads slower than the radio sends | "
        "Lower Sample Rate or lighten the chain. Raise Buffer Multiplier for short "
        "stalls. |\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Platform | Requirement |\n"
        "|---|---|\n"
        "| macOS | SoapySDR 0.8 and its modules, such as from Homebrew. |\n"
        "| Linux | SoapySDR 0.8 and its modules from the distribution packages. |\n"
        "| Windows | SoapySDR 0.8 and its modules, such as from PothosSDR, on the "
        "library search path. |\n"
        "| Browser, iOS, Android | Not available. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "A receive thread reads the radio into a ring buffer, and each cycle copies out"
        " one full buffer once enough samples have arrived. Frequency, Sample Rate, and"
        " gain changes are applied to the running stream, and Frequency and Sample Rate"
        " changes empty the ring buffer. The Bias-T is switched off again when the "
        "block closes the radio.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Radios with built-in drivers work without SoapySDR in **SDR**.\n"
        "- **Spectrum Analyzer** shows the band it receives.\n"
        "- **Filter** isolates one station from the capture.\n"
        "- **File Writer** records the samples to disk."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_BLOCK_HH
