#ifndef JETSTREAM_DOMAINS_DSP_AGC_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_AGC_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Agc : public Block::Config {
    U64 tileSize = 1024;
    F32 reference = 1.0f;
    F32 epsilon = 1e-12f;
    F32 minGain = 0.01f;
    F32 maxGain = 100.0f;
    F32 maxGainChange = 4.0f;

    JST_BLOCK_TYPE(agc);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_NODE_SIZE(XS);
    JST_BLOCK_PARAMS(tileSize, reference, epsilon, minGain, maxGain,
                     maxGainChange);
    JST_BLOCK_DESCRIPTION(
        "AGC",
        "Automatic Gain Control.",
        "# AGC\n"
        "The AGC block is a fixed-size tiled RMS feedforward AGC. It divides each "
        "independent sample lane into tiles, calculates one RMS gain per tile, and "
        "linearly interpolates adjacent gains while scaling the signal.\n\n"

        "## Arguments\n"
        "- **Tile Size**: Samples in each RMS tile; choose approximately sample rate "
        "times desired response time.\n"
        "- **Reference**: Desired RMS output level.\n"
        "- **Epsilon**: Power floor added before the square root.\n"
        "- **Min Gain / Max Gain**: Bounds applied to every tile gain.\n"
        "- **Max Gain Change**: Maximum multiplicative gain change between adjacent tiles.\n\n"

        "## Useful For\n"
        "- Stabilizing signal amplitude for consistent visualization.\n"
        "- Normalizing signals before further processing.\n"
        "- Compensating for varying input signal levels.\n\n"

        "## Examples\n"
        "- With Tile Size=1024, an F32[8192] lane uses eight RMS gains.\n"
        "- A final partial tile uses its actual sample count.\n\n"

        "## Implementation\n"
        "Input -> Tile RMS -> Gain Limits -> Interpolation -> Scaling -> Output\n"
        "1. Compute each tile's mean power from `x^2` or `I^2 + Q^2`.\n"
        "2. Calculate `gain = reference / sqrt(mean_power + epsilon)`.\n"
        "3. Clamp gain and limit its change from the preceding tile.\n"
        "4. Interpolate each tile's gain toward the next tile's gain.\n"
        "5. Scale samples with phase-preserving saturation to the finite output range.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_AGC_BLOCK_HH
