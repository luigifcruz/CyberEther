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
        "Displays an eye diagram of an oversampled signal.",
        "# Eye Diagram\n"
        "\n"
        "Overlays the signal two symbols at a time, so the openings between symbol "
        "transitions show as an eye. It usually follows an RRC Filter and an AGC, at "
        "the full sample rate before a PSK Demodulator.\n"
        "\n"
        "- **Samples per Symbol is not detected.** Set it to match the oversampling of "
        "the source.\n"
        "- **Nothing recovers the symbol clock.** Each window starts at a fixed sample,"
        " so a rate mismatch makes the eye drift.\n"
        "- **Each buffer draws one window.** It is the newest complete window, and a "
        "partial window carries into the next buffer.\n"
        "- **Controls work in a detached window.** Click the expand icon on the view to"
        " open it.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | One dimension, read in order across buffers. "
        "The buffer must be contiguous. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Samples per Symbol** | 8 | 1 to 256 | Input samples in one symbol. Must "
        "match the source. |\n"
        "| **Interpolation** | 4x (Linear) | 1x (None), or 2x, 4x, 8x, 16x (Linear) | "
        "Points drawn per input sample, joined by straight lines. Adds display points, "
        "not bandwidth. |\n"
        "| **Component** | In-Phase (Real) | In-Phase (Real), Quadrature (Imaginary) | "
        "Part of each complex sample to draw. Real input takes In-Phase (Real) only. |\n"
        "| **Amplitude Min** | -2 | Any, below Amplitude Max | Level at the bottom of "
        "the plot. Lower levels are clipped. |\n"
        "| **Amplitude Max** | 2 | Any, above Amplitude Min | Level at the top of the "
        "plot. Higher levels are clipped. |\n"
        "| **Persistence** | 256 traces | 1 or more, slider up to 4,096 | Traces it "
        "takes for the density to fade to half. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume 4,096 complex samples per buffer at 1 MS/s.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Standard eye | Defaults | 125,000 symbols per second | A trace of 65 points "
        "across 2 symbols, every 4.1 ms, with density fading to half in 1.05 seconds |\n"
        "| Raw samples only | Interpolation 1x (None) | 125,000 symbols per second | A "
        "trace of 17 points across 2 symbols |\n"
        "| Smooth traces | Interpolation 16x (Linear) | 125,000 symbols per second | A "
        "trace of 257 points across 2 symbols |\n"
        "| Quadrature rail | Component Quadrature (Imaginary) | 125,000 symbols per "
        "second | A trace of 65 points from the imaginary part |\n"
        "| Faster symbols | Samples per Symbol 4 | 250,000 symbols per second | A trace"
        " of 33 points across 2 symbols |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The eye drifts sideways or smears into a band | Samples per Symbol does not "
        "match the source | Set it to the sample rate over the symbol rate, such as 8 "
        "for 125,000 symbols per second at 1 MS/s. |\n"
        "| Traces flatten along the top or bottom edge | Levels sit past Amplitude Min "
        "or Max and are clipped | Widen the range, or put an AGC upstream. |\n"
        "| The block reports that real input supports only the real component | "
        "Component is Quadrature (Imaginary) with F32 input | Pick In-Phase (Real), or "
        "feed complex samples. |\n"
        "| The block reports an invalid input rank | The input has two or more "
        "dimensions, such as batches of heads from a Filter | Pick one head with a "
        "Slice block. |\n"
        "| The block reports an unsupported input data type | The input is neither F32 "
        "nor CF32 | Convert it with a Cast block. |\n"
        "\n"
        "## Controls\n"
        "\n"
        "| Action | Input |\n"
        "|---|---|\n"
        "| Zoom the time axis around the cursor, up to 10x | Scroll |\n"
        "| Pan while zoomed | Drag |\n"
        "| Reset zoom and pan | Right Click |\n"
        "| Freeze or resume the view | Space |\n"
        "| Place a marker, or remove the one under the cursor | Shift+Click |\n"
        "| Clear every marker | Shift+Right Click |\n"
        "| Move a marker | Drag its line |\n"
        "| Show the distances to neighboring markers | Hover a marker tag, click it to "
        "pin |\n"
        "| Unpin every tag | Escape |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The stream is cut into windows that share their end samples, and straight "
        "lines between samples add the display points. Levels are clipped to the "
        "amplitude range and scaled to the plot, and a density map fades older traces. "
        "The model below covers one buffer from a fresh start, without non-finite "
        "samples or the density.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def eye_diagram(x, samples_per_symbol=8, interpolation=4,\n"
        "                range_min=-2.0, range_max=2.0):\n"
        "    n = 2 * samples_per_symbol\n"
        "    windows = (len(x) - 1) // n\n"
        "    if windows == 0:\n"
        "        return np.full(n * interpolation + 1, 0.5)\n"
        "    w = x[(windows - 1) * n:windows * n + 1]\n"
        "    t = np.arange(n * interpolation + 1) / interpolation\n"
        "    y = np.interp(t, np.arange(n + 1), w)\n"
        "    return (np.clip(y, range_min, range_max) - range_min) / (range_max - range_min)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- Symbols recovered by the **PSK Demodulator** show on a **Constellation**.\n"
        "- Pulses shaped by an **RRC Filter** give the eye its opening.\n"
        "- Levels near 1 come from an **AGC** upstream.\n"
        "- **Lineplot** draws the same density view for a spectrum."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_BLOCK_HH
