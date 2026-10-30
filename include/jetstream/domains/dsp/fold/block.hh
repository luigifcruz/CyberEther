#ifndef JETSTREAM_DOMAINS_DSP_FOLD_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_FOLD_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Fold : public Block::Config {
    U64 offset = 0;
    U64 size = 0;

    JST_BLOCK_TYPE(fold);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS(offset, size);
    JST_BLOCK_DESCRIPTION(
        "Fold",
        "Folds the input signal along its sample axis.",
        "# Fold\n"
        "\n"
        "Shortens the sample axis to Size by wrapping it every Size samples and "
        "averaging the samples that land on each position. It usually sits between an "
        "FFT and an inverse FFT to decimate a spectrum, or averages the periods of a "
        "repeating signal.\n"
        "\n"
        "- **The default Size is rejected.** Enter a Size before the block can run.\n"
        "- **It is not a Decimator.** It averages samples spaced Size apart, not "
        "neighbors.\n"
        "- **The sample rate drops with the length.** The `sampleRate` attribute is "
        "divided by the same factor.\n"
        "- **Each lane is independent.** Every batch and channel is processed "
        "separately.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 or CF32 | Needs a `sampleAxis`, implied for "
        "one-dimensional input. A `batchAxis` and a `channelAxis` are optional. The "
        "buffer must be contiguous. |\n"
        "| **Output** (out) | Same as the input | The input shape with the sample axis "
        "set to Size. Every axis keeps its role. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Offset** | 0 samples | 0 up to the sample axis length | Moves every output"
        " value this many positions later, wrapping the end to the start. |\n"
        "| **Size** | 0 samples | Above 0, dividing the sample axis length | Output "
        "length along the sample axis. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Fold eight times | Size 1024 | 8,192 samples at 2 MS/s | 1,024 samples at "
        "250 kS/s, each the mean of 8 |\n"
        "| Average a repeating period | Size 100 | 1,000 samples holding 10 periods | "
        "100 samples, one averaged period |\n"
        "| Bring bin 1,000 to the start | Offset 7192, Size 1024 | 8,192 spectrum bins "
        "| 1,024 bins, with input bin 1,000 folded into index 0 |\n"
        "| Batched rows | Size 100 | 8 batches x 1,000 samples | 8 batches x 100 "
        "samples |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that Size cannot be zero | Size is 0, the default | Enter "
        "the output length, such as 1024. |\n"
        "| The block reports that Size is not a divisor of the input shape | The sample"
        " axis length is not a multiple of Size | Pick a Size that divides the length, "
        "or change the buffer size upstream. |\n"
        "| The block reports that the input needs valid signal axis metadata | The "
        "input has more than one dimension and no `sampleAxis` | Assign the roles with "
        "a Signal Axes block. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "| The block reports an unsupported data type | The input is not F32 or CF32 | "
        "Convert it with a Cast block. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block splits the sample axis into rows of Size samples and averages the "
        "rows, accumulating in double precision. Offset rotates the input before the "
        "split. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def fold(x, size, offset=0):\n"
        "    return np.roll(x, offset).reshape(-1, size).mean(axis=0)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Decimator** averages neighboring samples instead.\n"
        "- **Filter Engine** folds spectra to decimate each head.\n"
        "- **Signal Axes** assigns the `sampleAxis` it needs."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_FOLD_BLOCK_HH
