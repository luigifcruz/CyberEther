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
        "\n"
        "Converts integer samples to floating point scaled so signed full scale reads "
        "as 1, or turns real F32 samples into CF32. It usually follows a File Reader "
        "with raw samples, and feeds blocks that take only F32 or CF32.\n"
        "\n"
        "- **Real integers need two Casts for CF32.** Cast them to F32 first, then to "
        "CF32.\n"
        "- **Unsigned input is not centered.** Values are only scaled, so unsigned "
        "samples sit between 0 and 2.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | Real or complex integers of 8, 16, or 32 bits, F32, or CF32"
        " | Any shape with at least one axis. Strided input is accepted. |\n"
        "| **Output** (out) | Output Type | Same shape, with every axis and attribute "
        "kept. Integers are divided by 128, 32,768, or 2,147,483,648 for 8, 16, or 32 "
        "bits. An input already of Output Type passes through as the same buffer. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Output Type** | CF32 | CF32, F32 | CF32 takes complex integers or F32, and"
        " F32 takes real integers. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Raw complex file | Defaults | 8,192 CI8 samples | 8,192 complex samples from"
        " -1 to 0.99 |\n"
        "| Real 16-bit samples | F32 | 4,096 I16 samples | 4,096 real samples from -1 "
        "to 0.99997 |\n"
        "| Real to complex | Defaults | 8,192 real samples | 8,192 complex samples with"
        " zero imaginary parts |\n"
        "| Real integers to complex | F32, then a second Cast with CF32 | 4,096 I16 "
        "samples | 4,096 complex samples from -1 to 0.99997 |\n"
        "| Unsigned 8-bit file | Defaults | 8,192 CU8 samples | 8,192 complex samples "
        "from 0 to 1.99, centered near 1 |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports an unsupported conversion | Real integers with CF32, "
        "complex integers or complex floats with F32, or a type such as F64 | Pick F32 "
        "for real integers and CF32 for complex ones. Chain two Casts for real integers"
        " to CF32. |\n"
        "| A strong spike sits at 0 Hz after casting unsigned samples | Unsigned values"
        " are only scaled, so they center near 1 | Subtract the offset with a Python "
        "block. |\n"
        "| Levels peak far below 1 | The samples use fewer bits than their type, such "
        "as 12-bit values in I16 | Scale them up with Multiply Constant, such as 16 for"
        " 12-bit data. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block converts each value to floating point and scales integers so that "
        "signed full scale reads as one. Real F32 input gains a zero imaginary part, "
        "and a matching type is handed on without a copy. The model below covers one "
        "buffer with complex integers as real and imaginary pairs.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def cast(x, output_type=\"CF32\"):\n"
        "    if x.dtype == {\"F32\": np.float32, \"CF32\": np.complex64}[output_type]:\n"
        "        return x\n"
        "    if x.dtype == np.float32 and output_type == \"CF32\":\n"
        "        return x.astype(np.complex64)\n"
        "    y = (x / 2.0 ** (8 * x.itemsize - 1)).astype(np.float32)\n"
        "    if output_type == \"F32\":\n"
        "        return y\n"
        "    return (y[..., 0] + 1j * y[..., 1]).astype(np.complex64)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **File Reader** delivers the raw integer samples it converts.\n"
        "- **Multiply Constant** rescales samples that use fewer bits.\n"
        "- **Python** removes the offset of unsigned samples."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_CAST_BLOCK_HH
