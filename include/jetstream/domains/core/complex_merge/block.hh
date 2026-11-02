#ifndef JETSTREAM_DOMAINS_CORE_COMPLEX_MERGE_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_COMPLEX_MERGE_BLOCK_HH

#include <string>

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct ComplexMerge : public Block::Config {
    std::string mode = "rectangular";

    JST_BLOCK_TYPE(complex_merge);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(mode);
    JST_BLOCK_DESCRIPTION(
        "Complex Merge",
        "Merges two signals into a complex one.",
        "# Complex Merge\n"
        "\n"
        "Builds a complex signal from two real signals, either real and imaginary parts"
        " or magnitude and phase. It usually follows a Complex Split whose parts were "
        "processed separately, and feeds blocks that take CF32.\n"
        "\n"
        "- **The input labels follow Mode.** Real and Imag become Magnitude and Phase.\n"
        "- **Each lane is independent.** Every batch and channel is processed "
        "separately.\n"
        "- **Phase is in radians.** A phase of pi turns a sample around by half a "
        "circle.\n"
        "- **The Real input sets the attributes.** The output keeps every axis and "
        "attribute of the first input only.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Real** (in) | F32 | Any shape with at least one axis. Strided input is "
        "accepted. Labeled Magnitude in Magnitude and Phase. |\n"
        "| **Imag** (in) | F32 | Same shape as Real. Strided input is accepted. Labeled"
        " Phase in Magnitude and Phase. |\n"
        "| **Output** (out) | CF32 | Same shape, with every axis and attribute of Real "
        "kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Mode** | Real and Imag | Real and Imag, Magnitude and Phase | Real and "
        "Imag uses the inputs as the two parts. Magnitude and Phase turns each sample "
        "to the phase and scales it by the magnitude. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Rebuild after separate processing | Defaults | 8,192 real samples on each "
        "input | 8,192 complex samples |\n"
        "| Rebuild from polar parts | Magnitude and Phase | 8,192 magnitudes and 8,192 "
        "phases from a Complex Split | The original 8,192 complex samples |\n"
        "| Unit circle signal | Magnitude and Phase, Ones Tensor Shape [8192] on "
        "Magnitude | 8,192 phases | 8,192 complex samples of magnitude 1 |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the input shapes do not match | Real and Imag differ "
        "in shape | Feed both from buffers of the same shape. |\n"
        "| The block reports an unsupported input data type | An input is not F32, such"
        " as CF32 | Split complex input with Complex Split, or convert integers with a "
        "Cast block. |\n"
        "| The output reports the wrong `sampleRate` or `frequency` | Attributes come "
        "only from Real | Connect the signal whose attributes should be kept to Real. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each output sample pairs the values at the same position in both inputs. "
        "Magnitude and Phase converts each pair from polar to rectangular form. The "
        "model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def complex_merge(a, b, mode=\"Real and Imag\"):\n"
        "    if mode == \"Magnitude and Phase\":\n"
        "        return (a * np.cos(b) + 1j * a * np.sin(b)).astype(np.complex64)\n"
        "    return (a + 1j * b).astype(np.complex64)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Complex Split** separates a complex signal into two real ones.\n"
        "- **Cast** turns a real signal into complex with zero imaginary parts."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_COMPLEX_MERGE_BLOCK_HH
