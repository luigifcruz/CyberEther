#ifndef JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_BLOCK_HH
#define JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_BLOCK_HH

#include <string>

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct SignalAxes : public Block::Config {
    std::string axes;

    JST_BLOCK_TYPE(signal_axes);
    JST_BLOCK_DOMAIN("Core");
    JST_BLOCK_PARAMS(axes);
    JST_BLOCK_DESCRIPTION(
        "Signal Axes",
        "Assigns signal roles to tensor dimensions.",
        "# Signal Axes\n"
        "The Signal Axes block replaces the standard signal-axis metadata without "
        "copying or reordering tensor data. The output keeps the input shape, data "
        "type, device, strides, and all non-axis attributes.\n\n"

        "## Arguments\n"
        "- **Axes**: Positional axis roles in `[B, C, S, _, *]` notation. `B` marks "
        "the batch axis, `C` the channel axis, `S` the sample axis, and `_` a "
        "dimension with no explicit role. `*` preserves the input role currently "
        "assigned to that dimension. Each resulting role may appear at most once. "
        "Omitted trailing dimensions have no explicit role. Leave this field blank "
        "to validate and inherit every input role unchanged. An untagged rank-one "
        "tensor remains implicitly sampled by the standard signal-axis contract.\n\n"

        "## Useful For\n"
        "- Correcting or declaring signal metadata on imported tensors.\n"
        "- Marking a tensor as channel-only before visualization.\n"
        "- Clearing stale roles after a custom tensor transformation.\n\n"

        "## Examples\n"
        "- Channels followed by samples: `[C, S]`.\n"
        "- Antenna, channels, samples, polarization: `[_, C, S, _]`.\n"
        "- With input batch on axis 0, preserve it, clear channel, and assign "
        "samples: `[*, _, S]`.\n"
        "- Clear every explicit role on a rank-two tensor: `[_, _]`.\n\n"

        "## Implementation\n"
        "Input -> Metadata Overlay -> Output\n"
        "1. Parses and validates assigned and inherited roles against the input.\n"
        "2. Creates a zero-copy tensor view with independent metadata.\n"
        "3. Replaces only `batchAxis`, `channelAxis`, and `sampleAxis`.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_BLOCK_HH
