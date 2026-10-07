#ifndef JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_BLOCK_HH

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct MultiplyConstant : public Block::Config {
    F32 constant = 1.0f;

    JST_BLOCK_TYPE(multiply_constant);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(constant);
    JST_BLOCK_DESCRIPTION(
        "Multiply Constant",
        "Multiplies input by a constant value.",
        "# Multiply Constant\n"
        "\n"
        "Multiplies every value of the input by one real number, for a fixed gain, "
        "attenuation, or sign flip. It sits wherever a chain needs a fixed level "
        "change, such as between a demodulator and an Audio block.\n"
        "\n"
        "- **Constant is a linear factor.** A Constant of 10 adds 20 dB of gain, and "
        "0.5 halves the amplitude.\n"
        "- **Integer input is rejected.** Convert raw samples to F32 or CF32 with a "
        "Cast block first.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32, CF32, F64, or CF64 | Any shape. Strided input is "
        "accepted. |\n"
        "| **Output** (out) | Same as the input | Same shape, with every axis and "
        "attribute kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Constant** | 1 | Any | Factor applied to every value, and to both parts of"
        " complex values. The default leaves the input unchanged. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Double the amplitude | Constant 2 | 8,192 samples | 8,192 samples, 6 dB "
        "louder |\n"
        "| Attenuate by 20 dB | Constant 0.1 | 8,192 complex samples | 8,192 complex "
        "samples, 20 dB quieter |\n"
        "| Flip the sign | Constant -1 | 8 batches x 1,024 samples | 8 batches x 1,024 "
        "samples, inverted |\n"
        "| Lift Narrowband FM audio | Constant 4 | 8,192 samples | 8,192 samples, 12 dB"
        " louder |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports an unsupported data type | The input is an integer type, "
        "such as CI8 samples from a File Reader | Convert it with a Cast block. |\n"
        "| The level changes far more than expected | Constant was entered in decibels "
        "| Enter the linear factor, such as 10 for 20 dB of gain. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block scales each value by the constant into a new buffer, reading strided"
        " input in place. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "def multiply_constant(x, constant=1.0):\n"
        "    return x * constant\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Multiply** scales by a second tensor instead of one number.\n"
        "- Automatic level control comes from the **AGC**.\n"
        "- **Amplitude** reads the result in decibels."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_BLOCK_HH
