#ifndef JETSTREAM_DOMAINS_CORE_CAST_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_CAST_BLOCK_HH

#include <string>

#include "jetstream/block.hh"
#include "jetstream/types.hh"

namespace Jetstream::Blocks {

struct Cast : public Block::Config {
    std::string outputType = "CF32";

    JST_BLOCK_TYPE(cast);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(outputType);
    JST_BLOCK_DESCRIPTION(
        "Cast",
        "Casts the input to a type.",
        "# Cast\n"
        "The Cast block converts supported input tensors to the selected output "
        "type so downstream blocks receive data in the format they expect.\n\n"

        "## Arguments\n"
        "- **Output Type**: The desired output data type (e.g., CF32, F32).\n"

        "## Useful For\n"
        "- Normalizing integer-based real or complex data to floating-point for "
        "further processing.\n"
        "- Adapting data types between different processing stages in the "
        "flowgraph.\n"
        "- Preparing SDR samples for modules that expect F32 or CF32 input.\n\n"

        "## Examples\n"
        "- I16 normalization:\n"
        "  Config: OutputType=F32\n"
        "  Input: I16[4096] -> Output: F32[4096]\n"
        "- CI8 normalization:\n"
        "  Config: OutputType=CF32\n"
        "  Input: CI8[4096] -> Output: CF32[4096]\n\n"

        "## Implementation\n"
        "For each element, the input value or components are converted to the "
        "selected floating-point output type. Integer inputs are normalized by "
        "their default dtype scale.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_CAST_BLOCK_HH
