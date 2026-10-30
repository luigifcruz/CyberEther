#ifndef JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct EyeDiagram : public Block::Config {
    U64 samplesPerSymbol = 8;
    U64 interpolation = 4;
    std::string component = "real";
    F32 rangeMin = -2.0f;
    F32 rangeMax = 2.0f;
    U64 persistence = 256;

    JST_BLOCK_TYPE(eye_diagram);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(samplesPerSymbol, interpolation, component, rangeMin, rangeMax,
                     persistence);
    JST_BLOCK_DESCRIPTION(
        "Eye Diagram",
        "Displays a two-symbol eye diagram with linear interpolation.",
        "# Eye Diagram\n\n"
        "Folds a real signal, or the selected I/Q component of a complex signal, "
        "into two-symbol windows and displays them using the Lineplot density "
        "renderer. Connect an oversampled, pulse-shaped signal before symbol "
        "decimation.\n\n"
        "## Controls\n\n"
        "- **Samples per Symbol** must match the input sample rate divided by "
        "the symbol rate. Changing it resets the folding history.\n"
        "- **Interpolation** selects points per input interval: 1 disables "
        "interpolation; 2, 4, 8 and 16 use linear interpolation. This adds display "
        "points, not signal bandwidth or timing recovery.\n"
        "- **Component** selects in-phase (real) or quadrature (imaginary). "
        "Real input supports in-phase only.\n"
        "- **Amplitude Min/Max** set the visible linear amplitude range.\n"
        "- **Persistence** controls density fading in displayed traces.\n\n"
        "## Ports\n\n"
        "**Input:** a contiguous one-dimensional F32 or CF32 sample stream.\n\n"
        "Each input buffer displays its latest complete two-symbol window. "
        "Partial windows and the interpolation endpoint carry across buffers. "
        "Density accumulates successive displayed windows. The horizontal axis "
        "is normalized symbol time from -1 to +1. No clock recovery is applied; "
        "a rate mismatch will cause the eye to drift."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_BLOCK_HH
