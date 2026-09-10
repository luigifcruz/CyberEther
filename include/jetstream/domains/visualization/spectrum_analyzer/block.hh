#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SpectrumAnalyzer : public Block::Config {
    U64 lineplotAveraging = 1;
    U64 waterfallAveraging = 1;
    bool maxHold = false;
    bool fill = true;
    F32 rangeMin = -100.0f;
    F32 rangeMax = 0.0f;
    U64 waterfallHeight = 1024;
    F32 splitRatio = 0.5f;
    std::string xLabel = "Frequency (MHz)";
    std::string amplitudeLabel = "Amplitude (dBFS)";
    std::string waterfallLabel = "Time";

    JST_BLOCK_TYPE(spectrum_analyzer);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(lineplotAveraging, waterfallAveraging, maxHold,
                     fill, rangeMin, rangeMax, waterfallHeight,
                     splitRatio, xLabel, amplitudeLabel, waterfallLabel);
    JST_BLOCK_DESCRIPTION(
        "Spectrum Analyzer",
        "Spectrum trace and waterfall in one view.",
        "# Spectrum Analyzer\n"
        "The Spectrum Analyzer accepts complex samples, computes a normalized "
        "spectrum sized by the input sample axis, and renders a line trace above "
        "a scrolling waterfall on one surface. Both views share horizontal zoom "
        "and pan.\n\n"

        "The input must identify its sample dimension with `sampleAxis`. An "
        "optional `batchAxis` is averaged into the line trace and appended to "
        "the waterfall history. Channel and auxiliary dimensions are not "
        "supported.\n\n"

        "## Arguments\n"
        "- **Range Min/Max**: Reference bounds for the soft display mapping.\n"
        "- **Lineplot Averaging**: Exponential trace smoothing factor across updates, "
        "initialized from the first batch.\n"
        "- **Waterfall Averaging**: Number of spectra averaged per displayed row.\n"
        "- **Max Hold**: Retain the maximum observed trace.\n"
        "- **Waterfall Height**: Number of spectrum rows retained.\n\n"

        "Both averages operate on affine-normalized decibel values before soft "
        "display mapping. For ideal Gaussian noise above the numerical floor, log averaging "
        "has an expected bias of about -2.51 dB relative to mean power expressed "
        "in dB. This is not a universal offset for other signal statistics.\n\n"

        "## Implementation\n"
        "Complex Input -> Window -> FFT -> Amplitude -> Range -> Combined Plot\n"
        "The waterfall always uses smoothed interpolation and retains full-"
        "resolution spectrum rows.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH
