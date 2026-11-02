#ifndef JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_BLOCK_HH
#define JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct OverlapAdd : public Block::Config {
    JST_BLOCK_TYPE(overlap_add);
    JST_BLOCK_DOMAIN("DSP");
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Overlap Add",
        "Sums overlap with buffer for streaming convolution.",
        "# Overlap Add\n"
        "\n"
        "Adds the tail of each block of an FFT convolution to the start of the next "
        "block, so the filtered stream has no seams. It usually follows an Unpad that "
        "splits the inverse FFT into the result and its tail, in a hand-built FFT "
        "filter.\n"
        "\n"
        "- **Feed both inputs from one Unpad.** Its Output goes to Buffer and its Pad "
        "goes to Overlap.\n"
        "- **Batches form one stream.** Each batch receives the overlap of the one "
        "before, across buffers too.\n"
        "- **Channels keep separate state.** Each channel carries its own overlap into "
        "the next buffer.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Buffer** (in) | F32 or CF32 | Needs a `sampleAxis`, implied for "
        "one-dimensional input. A `batchAxis` and a `channelAxis` are optional. The "
        "buffer must be contiguous. |\n"
        "| **Overlap** (in) | Same as Buffer | Same rank, roles, and sizes as Buffer, "
        "except a sample axis no longer than Buffer's. The buffer must be contiguous. |\n"
        "| **Output** (out) | Same as Buffer | Same shape as Buffer, with every axis "
        "and attribute of Buffer kept. |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the buffer and overlap axes must match | The inputs "
        "place `sampleAxis`, `batchAxis`, or `channelAxis` differently | Feed both "
        "inputs from the same Unpad. |\n"
        "| The block reports that the overlap is larger than the buffer | The sample "
        "axis of Overlap is longer than that of Buffer | Lower Pad Size in the Unpad, "
        "or use longer buffers. |\n"
        "| The block reports a rank or shape mismatch | The inputs differ in rank, or "
        "in size on another axis | Feed both inputs from the same Unpad. |\n"
        "| The block reports a mismatched or unsupported data type | The inputs are not"
        " both F32 or both CF32 | Convert them with a Cast block. |\n"
        "| The block reports that it expects a contiguous tensor | An input is strided,"
        " such as a Slice with Contiguous off | Turn on Contiguous in the Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "The block copies Buffer to the output and adds the overlap of the previous "
        "batch to the first samples of each batch. The overlap of the last batch is "
        "saved for the next buffer, starting from zero. The model below covers one "
        "buffer with the saved overlap passed in and returned.\n"
        "\n"
        "```python\n"
        "def overlap_add(buffer, overlap, previous=0):\n"
        "    out = buffer.copy()\n"
        "    out[:len(overlap)] += previous\n"
        "    return out, overlap\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- **Unpad** splits the inverse FFT into Buffer and Overlap.\n"
        "- **Pad** makes room for the tail before the forward FFT.\n"
        "- **Filter** runs the whole overlap-add chain in one block."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_BLOCK_HH
