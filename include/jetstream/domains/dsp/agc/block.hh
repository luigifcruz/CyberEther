#ifndef JETSTREAM_DOMAINS_DSP_AGC_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_AGC_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Agc : public Block::Config {
    JST_BLOCK_TYPE(agc);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "AGC",
        "Automatic Gain Control.",
        "# AGC\n"
        "The AGC block automatically adjusts the gain of the input signal to "
        "maintain a constant output level. It normalizes the signal amplitude by "
        "finding the maximum value and scaling all elements to achieve a target "
        "level of 1.0.\n\n"

        "## Useful For\n"
        "- Stabilizing signal amplitude for consistent visualization.\n"
        "- Normalizing signals before further processing.\n"
        "- Compensating for varying input signal levels.\n\n"

        "## Examples\n"
        "- Complex signal normalization:\n"
        "  Input: CF32[1024] with max amplitude 0.5 -> Output: CF32[1024] with max amplitude 1.0\n"
        "- Real signal normalization:\n"
        "  Input: F32[1024] with max amplitude 2.0 -> Output: F32[1024] with max amplitude 1.0\n\n"

        "## Implementation\n"
        "Input -> Find Max -> Calculate Gain -> Apply Gain -> Output\n"
        "1. Find the maximum absolute value in the input buffer.\n"
        "2. Calculate gain as 1.0 / max_value.\n"
        "3. Multiply all elements by the calculated gain.\n"
        "4. Output has the same shape as input with normalized amplitude.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_AGC_BLOCK_HH
