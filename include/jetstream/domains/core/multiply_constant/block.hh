#ifndef JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct MultiplyConstant : public Block::Config {
    F32 constant = 1.0f;

    JST_BLOCK_TYPE(multiply_constant);
    JST_BLOCK_PARAMS(constant);
    JST_BLOCK_DESCRIPTION(
        "Multiply Constant",
        "Multiplies input by a constant value.",
        "# Multiply Constant\n"
        "The Multiply Constant block performs element-wise multiplication of an input tensor by a "
        "scalar constant value. This is useful for applying gain, attenuation, or scaling factors "
        "to signals without requiring a second input tensor.\n\n"

        "## Arguments\n"
        "- **Constant**: The scalar value to multiply each element by.\n\n"

        "## Useful For\n"
        "- Applying fixed gain or attenuation to signals.\n"
        "- Scaling signal levels for normalization.\n"
        "- Amplitude adjustment in signal processing chains.\n"
        "- Converting between units (e.g., linear to percentage).\n\n"

        "## Examples\n"
        "- Apply 2x gain:\n"
        "  Config: Constant=2.0\n"
        "  Input: F32[1024] -> Output: F32[1024] (each element doubled)\n"
        "- Apply 50% attenuation:\n"
        "  Config: Constant=0.5\n"
        "  Input: CF32[512] -> Output: CF32[512] (each element halved)\n\n"

        "## Implementation\n"
        "Input -> Multiply by Constant -> Output\n"
        "output[i] = input[i] * constant";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_BLOCK_HH
