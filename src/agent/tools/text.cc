#include "support/base.hh"

namespace Jetstream::Agent::Detail {
namespace {

Json ReadBlockTextTool(const Graph& flowgraph, const Json& args) {
    const std::string name = args.value("name", std::string{});
    const std::string field = args.value("field", std::string{});

    std::string text;
    if (const auto error = BlockText::Read(flowgraph, name, field, text); !error.empty()) {
        return Response::Error(error);
    }

    std::string error;
    auto lines = BlockText::Range(text, args, error);
    if (!error.empty()) {
        return Response::Error(error);
    }

    Json result = {
        {"name", name},
        {"field", field},
    };
    result.update(lines);
    return result;
}

}  // namespace

std::vector<Entry> Toolbox::Text() {
    return {
        Entry::FromJson({
            "read_block_text",
            "Read or search one text config field, such as python code, by line. "
            "Prefer this for a relevant excerpt instead of fetching the entire "
            "configuration with inspect_block. With pattern, returns matching "
            "lines (match=true) plus context; pattern is literal unless regex is "
            "true. Otherwise returns lines from offset (1-based) up to limit, with "
            "next_offset when more remain. Returned line numbers are separate "
            "metadata; never copy them into patches. Use patch_block_text in "
            "apply_flowgraph_patch for partial edits, or reconfigure_block to "
            "replace the entire field.",
            R"({"type":"object","required":["name","field"],"properties":{"name":{"type":"string","description":"Exact block name from inspect_flowgraph."},"field":{"type":"string","description":"Text config field, such as code."},"pattern":{"type":"string","description":"Text to find on a line."},"regex":{"type":"boolean","description":"Treat pattern as an ECMAScript regular expression."},"context":{"type":"integer","minimum":0,"maximum":20,"description":"Lines of context around each match. Default 2."},"offset":{"type":"integer","minimum":1,"description":"First line to return when pattern is absent. Default 1."},"limit":{"type":"integer","minimum":1,"maximum":500,"description":"Maximum lines to return. Default 200."}},"additionalProperties":false})",
            {.needsGraph = true},
        }, ReadBlockTextTool),
    };
}

}  // namespace Jetstream::Agent::Detail
