#ifndef JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SignalGenerator : public Block::Config {
    std::string signalType = "cosine";
    std::string signalDataType = "F32";
    F32 sampleRate = 1000000.0;
    F32 frequency = 1000.0;
    F32 amplitude = 1.0;
    F32 phase = 0.0;
    F32 dcOffset = 0.0;
    F32 noiseVariance = 1.0;
    F32 chirpStartFreq = 1000.0;
    F32 chirpEndFreq = 10000.0;
    F32 chirpDuration = 1.0;
    U64 bufferSize = 8192;

    JST_BLOCK_TYPE(signal_generator);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(signalType, signalDataType, sampleRate, frequency,
                     amplitude, phase, dcOffset, noiseVariance,
                     chirpStartFreq, chirpEndFreq, chirpDuration, bufferSize);
    JST_BLOCK_DESCRIPTION(
        "Signal Generator",
        "Generates synthetic waveforms, noise, and chirps.",
        "# Signal Generator\n"
        "\n"
        "Produces tones, periodic waves, noise, a constant level, or a repeating chirp,"
        " as real or complex samples. It usually starts a test chain in place of a "
        "radio source, feeding a Filter, a demodulator, or a Spectrum Analyzer.\n"
        "\n"
        "- **Settings follow Signal Type.** Only the settings that apply to the "
        "selected waveform are shown.\n"
        "- **Not every waveform is complex in CF32.** Square, Triangle, Sawtooth, and "
        "DC keep the imaginary part at 0.\n"
        "- **Sample Rate does not pace the output.** Buffers come out as fast as the "
        "chain runs, labeled with Sample Rate.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | F32 or CF32, set by Data Type | Buffer Size samples on a "
        "`sampleAxis`. Sets `sampleRate` to Sample Rate and `frequency` to 0. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Signal Type** | Cosine | Sine, Cosine, Square, Triangle, Sawtooth, Noise, "
        "DC, Chirp | Waveform to generate. It decides which settings below are shown. |\n"
        "| **Data Type** | F32 | F32, CF32 | Real or complex output. |\n"
        "| **Sample Rate** | 1 MHz | Above 0 | Rate the waveform is computed at, and "
        "the output `sampleRate`. |\n"
        "| **Frequency** | 0.001 MHz | 0 to half the Sample Rate. Negative in CF32 Sine"
        " and Cosine. | Tone frequency. Shown for Sine, Cosine, Square, Triangle, and "
        "Sawtooth. |\n"
        "| **Start Frequency** | 0.001 MHz | 0 to half the Sample Rate. Down to minus "
        "half in CF32. | Frequency at the start of each sweep. Chirp only. |\n"
        "| **End Frequency** | 0.01 MHz | Same as Start Frequency | Frequency at the "
        "end of each sweep. Chirp only. |\n"
        "| **Duration** | 1 sec | At least 1 / Sample Rate | Length of each sweep, "
        "which then restarts at Start Frequency without a phase jump. Chirp only. |\n"
        "| **Amplitude** | 1 | 0 or more | Peak of the waveform, or the scale of the "
        "noise. Labeled Level for DC, where Additional Offset adds to it. |\n"
        "| **Noise Variance** | 1 | 0 or more | Gaussian variance before Amplitude, "
        "applied to the real and imaginary parts separately. Noise only. |\n"
        "| **Phase** | 0 rad | Any | Starting phase. Hidden for Noise and DC. |\n"
        "| **DC Offset** | 0 | Any | Constant added to the real part only. Labeled "
        "Additional Offset for DC. |\n"
        "| **Buffer Size** | 8,192 samples | 1 or more | Samples per output buffer. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows keep the default Sample Rate of 1 MHz and Buffer Size of 8,192.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Real test tone | Defaults | 8,192 real samples, a 1 kHz cosine with 1,000 "
        "samples per cycle |\n"
        "| Tone below center | CF32, Frequency -0.1 MHz | 8,192 complex samples with "
        "one tone at -100 kHz |\n"
        "| Square wave | Square, Frequency 0.01 MHz | 8,192 real samples between -1 and"
        " 1, 100 samples per cycle |\n"
        "| Unit-power noise | Noise, CF32, Noise Variance 0.5 | 8,192 complex samples "
        "with a mean power of 1 |\n"
        "| Full-band sweep | Chirp, CF32, Start -0.5 MHz, End 0.5 MHz, Duration 0.01 "
        "sec | 8,192 complex samples per buffer, one sweep every 10,000 samples |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that a frequency must be within a range | A frequency is "
        "above half the Sample Rate, or negative where only CF32 Sine, Cosine, and "
        "Chirp allow it | Lower the frequency, raise Sample Rate, or switch to one of "
        "those CF32 waveforms. |\n"
        "| A tone appears twice, mirrored around 0 Hz | The output has no imaginary "
        "part, as with F32 or a CF32 Square, Triangle, or Sawtooth | Use CF32 with "
        "Sine, Cosine, or Chirp. |\n"
        "| Square, Triangle, or Sawtooth shows extra tones across the spectrum | "
        "Harmonics above half the Sample Rate alias, since nothing band-limits them | "
        "Lower Frequency or raise Sample Rate. |\n"
        "| Noise is 3 dB stronger in CF32 than in F32 | Noise Variance applies to the "
        "real and imaginary parts separately | Halve Noise Variance for the same total "
        "power. |\n"
        "| The block reports that the chirp duration must be at least one sample period"
        " | Duration is below 1 / Sample Rate | Raise Duration. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "A phase accumulator advances on every sample and carries over between buffers,"
        " so tones and chirps stay continuous. Each waveform is then scaled by "
        "Amplitude and shifted by DC Offset, and Noise comes from a Gaussian generator."
        " The model below has one function per waveform for one buffer of real samples,"
        " with chirp sweeps a whole number of samples long.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def phase_ramp(frequency=1e3, sample_rate=1e6, phase=0.0, size=8192):\n"
        "    return phase + 2 * np.pi * frequency / sample_rate * np.arange(size)\n"
        "\n"
        "def sine(p):\n"
        "    return np.sin(p)\n"
        "\n"
        "def cosine(p):\n"
        "    return np.cos(p)\n"
        "\n"
        "def square(p):\n"
        "    return np.where(p % (2 * np.pi) < np.pi, 1.0, -1.0)\n"
        "\n"
        "def sawtooth(p):\n"
        "    return (p % (2 * np.pi)) / np.pi - 1\n"
        "\n"
        "def triangle(p):\n"
        "    u = (p % (2 * np.pi)) / (2 * np.pi)\n"
        "    return np.where(u < 0.5, 4 * u - 1, 3 - 4 * u)\n"
        "\n"
        "def noise(variance=1.0, size=8192):\n"
        "    return np.sqrt(variance) * np.random.standard_normal(size)\n"
        "\n"
        "def dc(size=8192):\n"
        "    return np.ones(size)\n"
        "\n"
        "def chirp(start=1e3, end=1e4, duration=1.0, sample_rate=1e6, size=8192):\n"
        "    t = (np.arange(size) % round(duration * sample_rate)) / sample_rate\n"
        "    f = start + (end - start) / duration * (t + 0.5 / sample_rate)\n"
        "    steps = np.concatenate(([0.0], f[:-1]))\n"
        "    return np.cos(np.cumsum(2 * np.pi * steps / sample_rate))\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Add** sums two generators, such as a tone and noise.\n"
        "- **Throttle** limits how often new buffers come out.\n"
        "- **Spectrum Analyzer** shows what the block produces.\n"
        "- **Soapy SDR** replaces it with live radio samples."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_BLOCK_HH
