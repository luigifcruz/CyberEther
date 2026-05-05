#ifndef JETSTREAM_DOMAINS_VISUALIZATION_NOTE_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_NOTE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Note : public Block::Config {
    std::string content = "# Note\nWrite your **markdown** here.";

    JST_BLOCK_TYPE(note);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_PARAMS(content);
    JST_BLOCK_DESCRIPTION(
        "Note",
        "Displays formatted markdown text inside a node.",
        "# Note\n"
        "The Note block renders user-provided markdown content directly in the "
        "flowgraph node. It has no signal inputs or outputs and performs no "
        "computation.\n\n"

        "## Arguments\n"
        "- **Content**: Markdown text to display.\n\n"

        "## Useful For\n"
        "- Annotating flowgraph pipelines with documentation.\n"
        "- Adding visual labels or descriptions to groups of blocks.\n"
        "- Embedding instructions or status notes.\n\n"

        "## Examples\n"
        "- Add a title note:\n"
        "  Config: Content='# FM Receiver\\nThis pipeline demodulates FM radio.'\n\n"

        "## Implementation\n"
        "The block contains no modules. The content parameter is rendered as "
        "markdown inside the flowgraph node.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_NOTE_BLOCK_HH
