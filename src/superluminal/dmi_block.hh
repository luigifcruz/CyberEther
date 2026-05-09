#ifndef JETSTREAM_SUPERLUMINAL_DMI_BLOCK_HH
#define JETSTREAM_SUPERLUMINAL_DMI_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct DynamicTensorImport : public Block::Config {
    Tensor buffer;

    JST_BLOCK_TYPE(dynamic_tensor_import);
    JST_BLOCK_DOMAIN("Other");
    JST_BLOCK_PARAMS(buffer);
    JST_BLOCK_DESCRIPTION(
        "Dynamic Tensor Import",
        "Dynamically imports an external tensor.",
        "Imports an external tensor buffer into the flowgraph for processing. "
        "Used internally by Superluminal to bridge user-provided data buffers "
        "into the signal processing pipeline."
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_SUPERLUMINAL_DMI_BLOCK_HH
