#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SpectrumAnalyzer : public Block::Config {
    U64 averaging = 1;
    U64 decimation = 1;
    bool maxHold = false;
    bool fill = true;
    F32 rangeMin = -100.0f;
    F32 rangeMax = 0.0f;
    U64 waterfallHeight = 512;
    std::string xLabel = "Frequency (MHz)";
    std::string amplitudeLabel = "Amplitude (dBFS)";
    std::string waterfallLabel = "Time";

    JST_BLOCK_TYPE(spectrum_analyzer);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(averaging, decimation, maxHold, fill, rangeMin,
                     rangeMax, waterfallHeight, xLabel,
                     amplitudeLabel, waterfallLabel);
    JST_BLOCK_DESCRIPTION(
        "Spectrum Analyzer",
        "Spectrum trace and waterfall in one view.",
        "# Spectrum Analyzer\n"
        "The Spectrum Analyzer accepts complex samples, computes a normalized "
        "spectrum, and renders a line trace above a scrolling waterfall on one "
        "surface. Both views share horizontal zoom and pan.\n\n"

        "The input must identify its sample dimension with `sampleAxis`. An "
        "optional `batchAxis` is averaged into the line trace and appended to "
        "the waterfall history. Channel and auxiliary dimensions are not "
        "supported.\n\n"

        "## Arguments\n"
        "- **Range Min/Max**: Display range mapped to the analyzer color scale.\n"
        "- **Averaging**: Trace smoothing factor.\n"
        "- **Decimation**: Trace-only horizontal decimation factor.\n"
        "- **Max Hold**: Retain the maximum observed trace.\n"
        "- **Waterfall Height**: Number of spectrum rows retained.\n\n"

        "## Implementation\n"
        "Complex Input -> Window -> FFT -> Amplitude -> Range -> Combined Plot\n"
        "The waterfall always uses smoothed interpolation and retains full-"
        "resolution spectrum rows even when the trace is decimated.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SPECTRUM_ANALYZER_BLOCK_HH
