#ifndef JETSTREAM_DOMAINS_CORE_COMPARATOR_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_COMPARATOR_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Comparator : public Block::Config {
    U64 inputCount = 2;
    F32 tolerance = 1e-6f;

    JST_BLOCK_TYPE(comparator);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(inputCount, tolerance);

    JST_BLOCK_DESCRIPTION(
        "Comparator",
        "Compares inputs for numerical similarity.",
        "# Comparator\n"
        "The Comparator block checks whether two or more tensors on the same device "
        "match within a configurable absolute tolerance. The first input is the "
        "reference. Every additional input is compared against it element-wise. This "
        "is useful for verifying that different implementations of the same algorithm "
        "produce matching results.\n\n"

        "## Arguments\n"
        "- **Input Count**: Number of input tensors to compare (2 to 16).\n"
        "- **Tolerance**: Maximum allowed absolute difference for a PASS result.\n"
        "The block always outputs the error tensor and reports PASS or FAIL through "
        "the Match metric.\n\n"

        "## Outputs\n"
        "- **Error**: Per-element difference magnitude, maxed across non-reference inputs.\n\n"

        "## Useful For\n"
        "- Cross-checking CPU vs GPU outputs after copying them to a common device.\n"
        "- Validating refactors of DSP chains against a golden path.\n"
        "- Spotting numerical drift between algorithm variants.\n\n"

        "## Examples\n"
        "- Two matching F32 vectors:\n"
        "  Input 0: F32[1024] == Input 1: F32[1024] -> Error: F32[1024] zeros, Match: PASS\n"
        "- Complex residual magnitude:\n"
        "  Input 0/1: CF32[512] -> Error: F32[512] with |a - b| per sample\n\n"

        "## Implementation\n"
        "Input 0 (reference) + Input 1..N-1 -> Element-wise absolute difference -> Error + Metrics\n"
        "1. All inputs must share the same shape, data type, and device.\n"
        "2. For each sample, compute max |reference - input_i| across secondary inputs.\n"
        "3. Publish max/mean/MSE metrics and PASS/FAIL match state.\n"
        "4. Supported dtypes: F32, F64, CF32, CF64."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_COMPARATOR_BLOCK_HH
