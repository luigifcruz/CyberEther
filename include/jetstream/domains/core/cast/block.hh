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
        "The Cast block converts supported input tensors into floating-point "
        "representations for downstream processing. It can normalize real-valued "
        "inputs to F32 and complex integer inputs to CF32 using a configurable "
        "scaling factor.\n\n"

        "## Arguments\n"
        "- **Output Type**: The desired output data type (e.g., CF32, F32).\n"
        "- **Scaler**: Scaling factor applied during conversion. When set to 0, "
        "an appropriate default is chosen based on the input type (e.g., 128.0 "
        "for I8 and CI8, 32768.0 for I16 and CI16). Integer input values are "
        "divided by this factor to produce normalized floating-point output.\n\n"

        "## Useful For\n"
        "- Normalizing integer-based real or complex data to floating-point for "
        "further processing.\n"
        "- Adapting data types between different processing stages in the "
        "flowgraph.\n"
        "- Preparing SDR samples for modules that expect F32 or CF32 input.\n\n"

        "## Examples\n"
        "- I16 normalization:\n"
        "  Config: OutputType=F32, Scaler=32768.0\n"
        "  Input: I16[4096] -> Output: F32[4096]\n"
        "- CI8 normalization:\n"
        "  Config: OutputType=CF32, Scaler=128.0\n"
        "  Input: CI8[4096] -> Output: CF32[4096]\n"
        "- F32 passthrough cast:\n"
        "  Config: OutputType=F32, Scaler=0.0\n"
        "  Input: F32[4096] -> Output: F32[4096]\n\n"

        "## Implementation\n"
        "For each element, the input value or components are converted to the "
        "selected floating-point output type. Integer inputs are normalized by "
        "the scaler, while F32 input is forwarded directly when casting to F32.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_CAST_BLOCK_HH
