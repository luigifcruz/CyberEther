#ifndef BLUEPRINT_GAIN_BLOCK_HH
#define BLUEPRINT_GAIN_BLOCK_HH

#include <jetstream/block.hh>
#include <jetstream/types.hh>

namespace Jetstream::Blocks {

struct BlueprintGain : public Block::Config {
    F32 gain = 1.0f;

    JST_BLOCK_TYPE(blueprint_gain);
    JST_BLOCK_DOMAIN("Blueprint");
    JST_BLOCK_PARAMS(gain);
    JST_BLOCK_DESCRIPTION(
        "Blueprint Gain",
        "Multiplies each input sample by a gain value.",
        "# Blueprint Gain\n"
        "A minimal plugin block that demonstrates how to expose a block, create "
        "a module, and register CPU native work with CyberEther.\n\n"
        "## Arguments\n"
        "- **Gain**: Scalar multiplier applied to every input sample.\n\n"
        "## Inputs\n"
        "- **Signal**: F32 or CF32 input tensor.\n\n"
        "## Outputs\n"
        "- **Signal**: Tensor with the same shape and type as the input."
    );
};

}  // namespace Jetstream::Blocks

#endif  // BLUEPRINT_GAIN_BLOCK_HH
