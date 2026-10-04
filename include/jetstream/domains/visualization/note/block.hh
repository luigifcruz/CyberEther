#ifndef JETSTREAM_DOMAINS_VISUALIZATION_NOTE_BLOCK_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_NOTE_BLOCK_HH

#include "jetstream/block.hh"

namespace Jetstream::Blocks {

struct Note : public Block::Config {
    std::string content = "# Note\nWrite your **markdown** here.";

    JST_BLOCK_TYPE(note);
    JST_BLOCK_DOMAIN("Visualization");
    JST_BLOCK_NODE_SIZE(XL);
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
        "- Embedding instructions or status notes.\n"
        "- Showing live values published to the flowgraph environment.\n\n"

        "## Live Markdown\n"
        "Placeholders such as `${env.status.snr:.1f}` show the current value of "
        "a flowgraph environment field while the flowgraph runs.\n\n"
        "- **Paths**: Dots select nested fields and brackets index lists, with "
        "negative indices counting from the end.\n"
        "- **Format**: An optional Python format specification follows the "
        "colon. Values that are not available yet render as `--`.\n"
        "- **Repeated rows**: A line containing `[*]` repeats once per entry, "
        "which builds a table from a list.\n"
        "- **Stat tiles**: A fenced block tagged `stats` renders one tile per "
        "line, written as label, value, and an optional tone such as green or "
        "red, separated by pipes.\n\n"

        "## Examples\n"
        "- Add a title note:\n"
        "  Config: Content='# FM Receiver\\nThis pipeline demodulates FM radio.'\n\n"

        "## Implementation\n"
        "The block contains no modules. The content parameter is rendered as "
        "markdown inside the flowgraph node, after placeholders are expanded "
        "from the flowgraph environment.";
    );
};

}  // namespace Jetstream::Blocks

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_NOTE_BLOCK_HH
