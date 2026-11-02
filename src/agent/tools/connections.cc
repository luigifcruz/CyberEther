#include "support/base.hh"
#include "tools/text.hh"

namespace Jetstream::Agent::Detail {
namespace {

Json FindConnectionOptions(const Graph& flowgraph, const Json& args) {
    if (!flowgraph) {
        return Response::Error("No flowgraph is currently selected.");
    }

    const std::string source = args.value("source", std::string{});
    const std::string destination = args.value("destination", std::string{});

    Flowgraph::View::BlockData sourceData;
    if (flowgraph->view().block(source, sourceData) != Result::SUCCESS) {
        return Response::Error(Lookup::Missing(flowgraph, source, "Source block"));
    }

    Flowgraph::View::BlockData destinationData;
    if (flowgraph->view().block(destination, destinationData) != Result::SUCCESS) {
        return Response::Error(Lookup::Missing(flowgraph, destination, "Destination block"));
    }

    const bool singlePair = sourceData.interfaceOutputs.size() == 1 &&
                            destinationData.interfaceInputs.size() == 1;

    Json options = Json::array();
    for (const auto& output : sourceData.interfaceOutputs) {
        for (const auto& input : destinationData.interfaceInputs) {
            const bool sameName = Text::ToLower(output.name) == Text::ToLower(input.name);

            std::string confidence = "unknown";
            if (sameName) {
                confidence = "high";
            } else if (singlePair) {
                confidence = "medium";
            }

            Json option = {
                {"output", output.name},
                {"input", input.name},
                {"confidence", confidence},
            };
            if (sameName) {
                option["reason"] = "Port names match";
            }
            options.push_back(std::move(option));
        }
    }

    return Json({
        {"source", source},
        {"destination", destination},
        {"options", std::move(options)},
    });
}

}  // namespace

std::vector<Entry> Toolbox::Connections() {
    return {
        Entry::FromJson({
            "find_connection_options",
            "Find candidate output/input port pairs between two existing blocks before connecting them. Pairs list the active ports; confidence is high when the port names match, and semantic compatibility is only determined when connected.",
            R"({"type":"object","required":["source","destination"],"properties":{"source":{"type":"string"},"destination":{"type":"string"}},"additionalProperties":false})",
            {.needsGraph = true},
        }, FindConnectionOptions),
        Entry::FromJson({
            "validate_flowgraph",
            "Check the selected graph for errored or incomplete blocks and unconnected inputs. Returns every issue with its code. BLOCK_ERRORED and BLOCK_INCOMPLETE mean the block is not ready, and inspect_block shows why. Fix it by reconfiguring or connecting it with apply_flowgraph_patch. When an issue lists config_options, valid values come only from those options. INPUT_UNCONNECTED names an input with no source, and find_connection_options lists compatible sources.",
            R"({"type":"object","properties":{},"additionalProperties":false})",
            {.needsGraph = true},
        }, Validation::Check),
    };
}

}  // namespace Jetstream::Agent::Detail
