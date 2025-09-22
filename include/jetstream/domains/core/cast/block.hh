#ifndef JETSTREAM_DOMAINS_CORE_CAST_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_CAST_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Cast : public Block::Config {
    std::string outputType = "CF32";
    F32 scaler = 0.0f;

    JST_BLOCK_TYPE(cast);
    JST_BLOCK_PARAMS(outputType, scaler);
    JST_BLOCK_DESCRIPTION(
        "Cast",
        "Casts the input to a type.",
        "# Cast\n"
        "The Cast block converts input tensors from one numeric type to another, "
        "enabling seamless data type transformations within the processing "
        "pipeline. It normalizes integer-based complex data to floating-point "
        "range using a configurable scaling factor.\n\n"

        "## Arguments\n"
        "- **Output Type**: The desired output data type (e.g., CF32, F32).\n"
        "- **Scaler**: Scaling factor applied during conversion. When set to 0, "
        "an appropriate default is chosen based on the input type (e.g., 128.0 "
        "for CI8, 32768.0 for CI16). The input values are divided by this "
        "factor to produce normalized floating-point output.\n\n"

        "## Useful For\n"
        "- Normalizing integer-based complex data to floating-point for further "
        "processing.\n"
        "- Adapting data types between different processing stages in the "
        "flowgraph.\n"
        "- Preparing SDR samples for signal processing modules that expect "
        "CF32 input.\n\n"

        "## Examples\n"
        "- CI8 normalization:\n"
        "  Config: OutputType=CF32, Scaler=128.0\n"
        "  Input: CI8[4096] -> Output: CF32[4096]\n"
        "- CI16 normalization:\n"
        "  Config: OutputType=CF32, Scaler=32768.0\n"
        "  Input: CI16[4096] -> Output: CF32[4096]\n\n"

        "## Implementation\n"
        "For each element, the input components are converted to the output "
        "type and divided by the scaler for normalization.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_CAST_BLOCK_HH
