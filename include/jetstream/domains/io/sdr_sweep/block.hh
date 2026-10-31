#ifndef JETSTREAM_DOMAINS_IO_SDR_SWEEP_BLOCK_HH
#define JETSTREAM_DOMAINS_IO_SDR_SWEEP_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SdrSweep : public Block::Config {
    std::string deviceString = "";
    std::string antenna = "";
    bool automaticGain = false;
    F32 manualGain = 30.0f;
    bool biasTee = false;
    F64 startFrequency = 88.0e6;
    F64 endFrequency = 108.0e6;
    F64 sampleRate = 0.0;
    F64 resolution = 10.0e3;
    U64 dwell = 1;
    F32 settlingTime = 0.0f;

    JST_BLOCK_TYPE(sdr_sweep);
    JST_BLOCK_DOMAIN("IO");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(deviceString, antenna, automaticGain, manualGain, biasTee,
                     startFrequency, endFrequency, sampleRate, resolution,
                     dwell, settlingTime);
    JST_BLOCK_DESCRIPTION(
        "SDR Sweep",
        "Sweeps a radio across a wide band.",
        "# SDR Sweep\n"
        "\n"
        "Retunes a radio across a band wider than its sample rate and outputs complex "
        "samples whose spectrum is the stitched band. It usually feeds a Spectrum "
        "Analyzer that shows the whole band at once.\n"
        "\n"
        "- **Each sweep is one buffer.** A buffer comes out only after every hop of a "
        "sweep was captured.\n"
        "- **Hops are captured at different times.** Phase is not continuous from one "
        "hop to the next.\n"
        "- **Labels describe the stitched band.** The `frequency` attribute is its "
        "center and `sampleRate` is its width.\n"
        "- **Automatic Gain is off by default.** Manual Gain keeps the levels of "
        "different hops comparable.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | CF32 | One batch x one sample per bin, on a `batchAxis` "
        "then a `sampleAxis`. Sets `frequency` to the band center and `sampleRate` to "
        "the band width, and adds `sweep` attributes with the grid and sweep number. "
        "The `sweepSynthetic` attribute tells the Spectrum Analyzer to skip its own "
        "window. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Device** | None | None and the radios found | Radio to sweep, saved by "
        "driver and serial number. A saved radio that is missing shows as Configured "
        "device. |\n"
        "| **Antenna** | Default | Default and the radio's receive ports | Receive "
        "port. Only LimeSDR lists ports. |\n"
        "| **Start Frequency** | 88 MHz | Within the radio's range | Lower edge of the "
        "band. Firmware sweeps round it down to a whole MHz. |\n"
        "| **End Frequency** | 108 MHz | Above Start Frequency | Upper edge. Firmware "
        "sweeps round the band up to whole 20 MHz steps. |\n"
        "| **Resolution** | 10 kHz | Above 0 | Target bin width. The grid can differ "
        "slightly, and coarsens only as far as capture and memory limits need. |\n"
        "| **Capture Rate** | Preferred | 0, or within the radio's range | Sample rate of "
        "each hop. Zero picks the radio's preferred rate, and Airspy, Airspy HF+, "
        "and BirriaSDR show a list. "
        "HackRF firmware sweeps need 0 or 20 MHz. |\n"
        "| **Dwell** | 1 | 1 to 64 | Captures averaged in power per hop, which smooths "
        "the noise floor. |\n"
        "| **Settling Time** | 0 ms | 0 or more | Time skipped after each retune. Zero "
        "uses the radio's default. A nonzero value makes HackRF retune instead of "
        "sweeping in firmware. |\n"
        "| **Automatic Gain** | Off | On, Off | Lets the radio set its gain, which "
        "makes hop levels not comparable. |\n"
        "| **Manual Gain** | 30 dB | Within the radio's gain range | Receive gain, shown "
        "when Automatic Gain is off. The value snaps to the nearest gain the radio "
        "applies at the start of the sweep. |\n"
        "| **Bias-T** | Off | On, Off | Powers an active antenna through the cable on "
        "RTL-SDR, Airspy, HackRF, and bladeRF 2. Hidden on other radios. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Buffer Health** | How often a finished sweep waits unread, from 0 to 100%."
        " Near 100% means the chain is falling behind. |\n"
        "| **Buffer Loss** | Share of finished sweeps replaced or expired before the "
        "chain read them. |\n"
        "| **Sweep Speed** | Band covered per second in GHz/s, followed by complete "
        "sweeps per second in Hz. Both are smoothed. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "Rows without a named radio assume an RTL-SDR at its preferred 2.4 MS/s. The "
        "rate is the `sampleRate` label, and sweeps arrive as often as Sweep Speed "
        "shows.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Broadcast FM band | Defaults | 2,000 complex samples at 20 MS/s per sweep, "
        "10 kHz bins from 88 to 108 MHz, 23 hops |\n"
        "| Land mobile radio | Start Frequency 400 MHz, End Frequency 470 MHz | 7,000 "
        "complex samples at 70 MS/s per sweep, 10 kHz bins, 81 hops |\n"
        "| Wi-Fi band on a HackRF | Start Frequency 2,400 MHz, End Frequency 2,500 MHz "
        "| 10,020 complex samples at 100 MS/s per sweep, 9.98 kHz bins, 10 firmware "
        "hops captured at 20 MS/s |\n"
        "| Calmer noise floor | Dwell 4 | Same as the first row, with 4 captures "
        "averaged per hop |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| Output bins are coarser than requested | The grid exceeds a capture or "
        "output limit | Narrow the span. |\n"
        "| The block reports that a hop is outside the tunable range | The band reaches"
        " past the range of the radio | Move the band inside the radio's range. |\n"
        "| Hop edges show traces of a neighboring hop | The radio settles slower than "
        "its default | Raise Settling Time by a few milliseconds. |\n"
        "| Buffer Loss rises above 0% | The chain reads slower than sweeps finish | "
        "Lighten the chain downstream. |\n"
        "| The sweep stops with an error | Hops failed three times in a row, often a "
        "loose cable | Check the radio connection and use Reload Block. |\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Platform | Requirement |\n"
        "|---|---|\n"
        "| macOS, Linux, Windows | Built-in drivers for RTL-SDR, Airspy, Airspy HF+, "
        "HackRF, LimeSDR, bladeRF, and PlutoSDR. Windows needs the WinUSB driver. |\n"
        "| BirriaSDR | macOS and Linux only, on a USB 3 port. Unbind the Wi-Fi driver, "
        "then replug the dongle. Each hop takes up to about 0.75 s. |\n"
        "| HackRF firmware | Firmware sweeps need USB API 0x0104 or newer. |\n"
        "| Browser | The same radios except PlutoSDR and BirriaSDR over WebUSB, once the "
        "page has access to the radio. |\n"
        "| iOS, Android | Not available. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Every capture gets a Blackman window and a transform, and Dwell averages the "
        "captures of each hop in power. Retuned hops overlap and blend in power with "
        "less weight near capture edges, while HackRF firmware hops use fixed bands "
        "that never overlap. An inverse transform turns the blended spectrum back into "
        "samples, keeping the phase of the hop that weighs most at each frequency.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Spectrum Analyzer** displays the stitched band.\n"
        "- A single band at full time resolution comes from **SDR**."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_IO_SDR_SWEEP_BLOCK_HH
