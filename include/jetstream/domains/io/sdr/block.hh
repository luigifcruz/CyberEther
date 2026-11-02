#ifndef JETSTREAM_DOMAINS_IO_SDR_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_SDR_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Sdr : public Block::Config {
    std::string deviceString = "";
    std::string antenna = "";
    F32 frequency = 96.9e6;
    F32 frequencyStep = 1000000.0;
    F32 sampleRate = 2.0e6;
    bool automaticGain = true;
    F32 manualGain = 30.0f;
    bool biasTee = false;
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_BLOCK_TYPE(sdr);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(deviceString, antenna, frequency, frequencyStep,
                     sampleRate, automaticGain, manualGain, biasTee, numberOfBatches,
                     numberOfTimeSamples, bufferMultiplier);
    JST_BLOCK_DESCRIPTION(
        "SDR",
        "Receives samples from a USB radio.",
        "# SDR\n"
        "\n"
        "Streams complex samples from a radio plugged in over USB, using drivers built "
        "into CyberEther. It usually starts a chain, feeding a Spectrum Analyzer to "
        "watch the band or a Filter that isolates one station.\n"
        "\n"
        "- **The radio paces the chain.** Each buffer comes out only once the radio has"
        " delivered all of its samples.\n"
        "- **Changes drop older samples.** Samples captured before a change never carry"
        " its new labels.\n"
        "- **A slow chain loses samples.** When the chain falls behind, the oldest "
        "samples are overwritten and Buffer Loss rises.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | CF32 | Batches x Samples, on a `batchAxis` then a "
        "`sampleAxis`. Sets `frequency` and `sampleRate` to the values the radio "
        "applied to these samples. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Device** | None | None and the radios found | Radio to open, saved by "
        "driver and serial number. A saved radio that is missing shows as Configured "
        "device. |\n"
        "| **Antenna** | Default | Default and the radio's receive ports | Receive "
        "port. Only LimeSDR lists ports. |\n"
        "| **Frequency** | 96.9 MHz | Within the radio's range | Center frequency of "
        "the output. |\n"
        "| **Sample Rate** | 2 MHz | Within the radio's range | Complex samples per "
        "second, which is also the bandwidth captured. Airspy and Airspy HF+ show a "
        "list of their rates, and a rate missing from the list switches to the first. |\n"
        "| **Automatic Gain** | On | On, Off | Lets the radio set its gain. HackRF, "
        "LimeSDR, and bladeRF 1 have none and always use Manual Gain. |\n"
        "| **Manual Gain** | 30 dB | Within the radio's gain range | Receive gain, "
        "shown when Automatic Gain is off. The value snaps to the nearest gain the "
        "radio applies, and on bladeRF 2 and PlutoSDR the limits follow the tuned "
        "frequency. |\n"
        "| **Bias-T** | Off | On, Off | Powers an active antenna through the cable on "
        "RTL-SDR, Airspy, HackRF, and bladeRF 2. Hidden on other radios. |\n"
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
        "since the stream started. Gaps the radio reported show next to it as GAP. |\n"
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
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that no device is selected | Device is None, the default |"
        " Pick a radio in Device. |\n"
        "| The block reports that the selected device is not connected | The radio is "
        "unplugged, or the browser has no access to it | Plug it in, then use Reload "
        "Block to refresh the Device list. In a browser, see Setup. |\n"
        "| The log warns that the frequency or sample rate is not supported | The value"
        " is outside the range of the radio | Pick a value the radio supports. The old "
        "value stays while running. |\n"
        "| Buffer Loss rises above 0% | The chain reads slower than the radio sends | "
        "Lower Sample Rate or lighten the chain. Raise Buffer Multiplier for short "
        "stalls. |\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Platform | Requirement |\n"
        "|---|---|\n"
        "| macOS, Linux, Windows | Built-in drivers for RTL-SDR, Airspy, Airspy HF+, "
        "HackRF, LimeSDR, bladeRF, and PlutoSDR. Windows needs the WinUSB driver for "
        "the radio. |\n"
        "| Browser | The same radios except PlutoSDR over WebUSB, once the page has "
        "been given access to the radio. |\n"
        "| iOS, Android | Not available. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "A receive thread reads the radio into a ring buffer, and each cycle copies out"
        " one full buffer. Frequency, Automatic Gain, and Manual Gain changes apply "
        "while the stream keeps running. A sample rate change reapplies the whole "
        "configuration.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Bands wider than the sample rate need **SDR Sweep**.\n"
        "- Radios with only a SoapySDR driver work through **SoapySDR**.\n"
        "- **Spectrum Analyzer** shows the band it receives."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_SDR_BLOCK_HH
