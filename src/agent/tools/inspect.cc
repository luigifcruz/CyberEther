#include "support/base.hh"
#include "jetstream/module_surface.hh"
#include "jetstream/runtime_context.hh"

#include <set>
#include <unordered_map>

namespace Jetstream::Agent::Detail {
namespace {

using BlockData = Flowgraph::View::BlockData;

constexpr U64 kConsoleLines = 50;

struct RouteEdge {
    std::string source;
    std::string output;
    std::string destination;
    std::string input;
};

bool SameExternalLink(const auto& a, const auto& b) {
    if (a.external.has_value() != b.external.has_value()) {
        return false;
    }
    if (!a.external.has_value()) {
        return true;
    }
    return a.external->block == b.external->block && a.external->port == b.external->port;
}

bool SameInputConnections(const BlockData& a, const BlockData& b) {
    if (a.inputs.size() != b.inputs.size()) {
        return false;
    }

    for (const auto& [port, link] : a.inputs) {
        const auto other = b.inputs.find(port);
        if (other == b.inputs.end() || !SameExternalLink(link, other->second)) {
            return false;
        }
    }

    return true;
}

Json RouteBlockJson(const Graph& flowgraph, const std::string& name, const BlockData& data) {
    Json block = {
        {"name", name},
        {"type", data.type},
        {"title", data.title},
        {"config", Serialize::CompactMap(data.config)},
        {"state", GetBlockStateName(data.state)},
    };
    block.update(Serialize::NodeLayout(flowgraph, name));

    if (!data.diagnostic.empty()) {
        block["diagnostic"] = data.diagnostic;
    }

    return block;
}

Json AlternativeJson(const RouteEdge& edge,
                     const BlockData& current,
                     const std::string& candidateName,
                     const BlockData& candidate) {
    return {
        {"replaces", edge.source},
        {"candidate", candidateName},
        {"type", candidate.type},
        {"current_config", Serialize::CompactMap(current.config)},
        {"candidate_config", Serialize::CompactMap(candidate.config)},
        {"shared_inputs", Serialize::Inputs(candidate)},
        {"switch_patch", {
            {"op", "connect"},
            {"source", candidateName},
            {"output", edge.output},
            {"block", edge.destination},
            {"input", edge.input},
        }},
    };
}

Json FlowgraphRoutingJson(const Graph& flowgraph, const std::string& sink) {
    std::vector<BlockData> entries;
    if (Serialize::Collect(flowgraph, entries) != Result::SUCCESS) {
        return {{"error", "Could not inspect the selected flowgraph."}};
    }

    std::vector<std::string> names;
    std::unordered_map<std::string, BlockData> blocks;
    for (auto& data : entries) {
        names.push_back(data.name);
        blocks.emplace(names.back(), std::move(data));
    }

    if (!blocks.contains(sink)) {
        return {
            {"error", jst::fmt::format("Sink block '{}' not found.", sink)},
            {"available_blocks", names},
        };
    }

    std::vector<RouteEdge> edges;
    std::set<std::string> active;
    std::function<void(const std::string&)> trace = [&](const std::string& blockName) {
        if (!active.insert(blockName).second) {
            return;
        }

        const auto block = blocks.find(blockName);
        if (block == blocks.end()) {
            return;
        }

        for (const auto& [input, link] : block->second.inputs) {
            if (!link.external.has_value()) {
                continue;
            }
            edges.push_back({link.external->block, link.external->port, blockName, input});
            trace(link.external->block);
        }
    };
    trace(sink);

    Json routeBlocks = Json::array();
    for (const auto& name : active) {
        routeBlocks.push_back(RouteBlockJson(flowgraph, name, blocks.at(name)));
    }

    std::unordered_map<std::string, Json> configs;
    const auto encodedConfig = [&](const std::string& name) -> const Json& {
        const auto found = configs.find(name);
        if (found != configs.end()) {
            return found->second;
        }
        return configs.emplace(name, Serialize::Map(blocks.at(name).config)).first->second;
    };

    Json connections = Json::array();
    Json alternatives = Json::array();
    for (const auto& edge : edges) {
        connections.push_back(Serialize::Connection(edge.source, edge.output, edge.destination, edge.input));

        const auto activeSource = blocks.find(edge.source);
        if (activeSource == blocks.end()) {
            continue;
        }
        const auto& current = activeSource->second;

        for (const auto& [candidateName, candidate] : blocks) {
            if (active.contains(candidateName) || candidate.type != current.type) {
                continue;
            }
            if (!SameInputConnections(current, candidate)) {
                continue;
            }
            if (encodedConfig(edge.source) == encodedConfig(candidateName)) {
                continue;
            }
            if (!Serialize::HasPort(candidate.interfaceOutputs, edge.output)) {
                continue;
            }

            alternatives.push_back(AlternativeJson(edge, current, candidateName, candidate));
        }
    }

    return {
        {"sink", sink},
        {"active_route", {
            {"blocks", std::move(routeBlocks)},
            {"connections", std::move(connections)},
        }},
        {"alternative_branches", std::move(alternatives)},
    };
}

Json InspectFlowgraph(const Graph& flowgraph, const Json& args) {
    if (!flowgraph) {
        return Response::Error("No flowgraph is selected.");
    }

    const std::string detail = args.value("detail", std::string("summary"));
    if (detail != "routing") {
        return Serialize::FlowgraphDetail(flowgraph, detail == "full");
    }

    const std::string sink = args.value("sink", std::string{});
    if (sink.empty()) {
        return Response::Error("'sink' is required for routing detail.");
    }

    return FlowgraphRoutingJson(flowgraph, sink);
}

Json DiagnosticJson(const Runtime::Context::Diagnostic& diagnostic) {
    const auto& console = diagnostic.console;
    const U64 first = console.size() > kConsoleLines ? console.size() - kConsoleLines : 0;

    Json value = {
        {"healthy", diagnostic.healthy},
        {"status", diagnostic.status},
        {"console", Json(std::vector<std::string>(console.begin() + first, console.end()))},
    };
    if (first > 0) {
        value["console_lines"] = console.size();
    }

    return value;
}

Json MetricJson(const Flowgraph::View::MetricEntry& metric) {
    Json value;
    if (const auto* diagnostic = std::any_cast<Runtime::Context::Diagnostic>(&metric.value)) {
        value = DiagnosticJson(*diagnostic);
    } else if (!Value::TryToJson(metric.value, value)) {
        value = Value::AnyToString(metric.value);
    }

    Json entry = {
        {"name", metric.name},
        {"label", metric.label},
        {"value", std::move(value)},
    };
    if (!metric.format.empty()) {
        entry["format"] = Serialize::Map(metric.format);
    }
    if (!metric.help.empty()) {
        entry["help"] = metric.help;
    }

    return entry;
}

Json InspectBlock(const Graph& flowgraph, const Json& args) {
    if (!flowgraph) {
        return Response::Error("No flowgraph is currently selected.");
    }

    const std::string name = args.value("name", std::string{});
    Flowgraph::View::BlockData data;
    if (name.empty() || flowgraph->view().block(name, data) != Result::SUCCESS) {
        return Response::Error(Lookup::Missing(flowgraph, name));
    }

    Json result = Serialize::BlockDetail(flowgraph, data, true);

    Json metrics = Json::array();
    for (const auto& metric : data.metrics) {
        metrics.push_back(MetricJson(metric));
    }
    result["metrics"] = std::move(metrics);

    Json surfaces = Json::array();
    for (const auto& surface : data.surfaces) {
        for (const auto& manifest : surface->manifests()) {
            surfaces.push_back(manifest.id);
        }
    }
    result["surfaces"] = std::move(surfaces);

    return result;
}

}  // namespace

std::vector<Entry> Toolbox::Inspect() {
    auto flowgraph = Entry::FromJson({
            "inspect_flowgraph",
            "Inspect the selected graph. Summary and routing shorten long strings "
            "and arrays in effective configs. Summary lists "
            "each block's connections under its inputs as the upstream "
            "block and port. Summary omits device, runtime, and provider when they "
            "are cpu, native, and generic. Use routing with a sink block to trace "
            "its active upstream path and discover exact alternative branches. "
            "Each alternative carries a switch_patch operation that switches the "
            "sink to that branch when sent through apply_flowgraph_patch. Full "
            "includes complete effective configs and active interfaces but omits "
            "repeated block documentation. Blocks include saved node settings and "
            "measured outer bounds in graph coordinates when layout_status is "
            "'measured'; bounds is null while 'pending'. node omits width and height "
            "while the node uses automatic sizing. When the graph is on screen, "
            "viewport gives the visible canvas rectangle, the toolbar rectangle "
            "floating over its top edge, and the open_area below the toolbar, "
            "all in graph coordinates.",
            R"({"type":"object","properties":{"detail":{"type":"string","enum":["summary","routing","full"]},"sink":{"type":"string","description":"For routing detail, the downstream block to trace, such as an audio or visualization sink."}},"additionalProperties":false})",
            {.needsGraph = true, .reportsLayout = true},
        }, InspectFlowgraph);
    flowgraph.annotate = [](Json& body, const Json&, const Layout& layout) { Geometry::Viewport(body, layout); };
    return {
        std::move(flowgraph),
        Entry::FromJson({
            "inspect_block",
            "Inspect one existing block: status, complete effective JSON "
            "configuration, interfaces, connections, metrics, node metadata, and "
            "rendered surfaces. Interfaces omit help text, which get_block_docs "
            "provides. node omits width and height while the node uses automatic "
            "sizing. bounds gives the measured outer rectangle in graph coordinates "
            "when layout_status is 'measured', or null while 'pending'. "
            "Use bounds for positioning without overlap. Text and code nodes "
            "report content_size, the inner width and height that show all of "
            "their content, and clipped=true when the node is smaller.",
            R"({"type":"object","required":["name"],"properties":{"name":{"type":"string","description":"Exact block name from inspect_flowgraph."}},"additionalProperties":false})",
            {.needsGraph = true, .reportsLayout = true},
        }, InspectBlock),
    };
}

}  // namespace Jetstream::Agent::Detail
