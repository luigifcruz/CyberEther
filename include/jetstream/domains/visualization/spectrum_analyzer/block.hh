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
        "- **Range Min/Max**: Display range mapped to the analyzer color scale.\n"
        "- **Lineplot Averaging**: Trace smoothing factor.\n"
        "- **Waterfall Averaging**: Number of spectra averaged per displayed row.\n"
        "- **Max Hold**: Retain the maximum observed trace.\n"
        "- **Waterfall Height**: Number of spectrum rows retained.\n\n"

        "Both averages operate on the decibel spectrum, matching the log "
        "averaging that bench spectrum analyzers use by default. The noise "
        "floor therefore reads about 2.5 dB below its true mean power.\n\n"

        "## Implementation\n"
        "Complex Input -> Window -> FFT -> Amplitude -> Range -> Combined Plot\n"
        "The waterfall always uses smoothed interpolation and retains full-"
        "resolution spectrum rows.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH
