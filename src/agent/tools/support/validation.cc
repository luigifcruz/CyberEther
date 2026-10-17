#include "base.hh"

namespace Jetstream::Agent::Detail {
namespace {

bool InputConnected(const Flowgraph::View::BlockData& data, const std::string& name) {
    bool connected = false;
    for (const auto& [port, link] : data.inputs) {
        if (port == name && link.external.has_value()) {
            connected = true;
        }
    }
    return connected;
}

}  // namespace

Json Validation::Check(const Graph& flowgraph, const Json&) {
    if (!flowgraph) {
        return Response::Error("No flowgraph is currently selected.");
    }

    std::vector<Flowgraph::View::BlockData> blocks;
    if (Serialize::Collect(flowgraph, blocks) != Result::SUCCESS) {
        return Response::Error("Could not inspect the flowgraph.");
    }

    Json issues = Json::array();
    for (const auto& data : blocks) {
        if (auto options = Serialize::PendingConfigOptions(data)) {
            const bool errored = data.state == Block::State::Errored;
            const std::string message = data.diagnostic.empty() ? "Block is not ready" : data.diagnostic;

            Json issue = {
                {"severity", "error"},
                {"code", errored ? "BLOCK_ERRORED" : "BLOCK_INCOMPLETE"},
                {"block", data.name},
                {"message", message},
            };
            if (!options->empty()) {
                issue["config_options"] = std::move(*options);
            }
            issues.push_back(std::move(issue));
        }

        for (const auto& input : data.interfaceInputs) {
            if (InputConnected(data, input.name)) {
                continue;
            }

            issues.push_back({
                {"severity", "warning"},
                {"code", "INPUT_UNCONNECTED"},
                {"block", data.name},
                {"input", input.name},
            });
        }
    }

    return Validation::Health(issues, issues.size());
}

std::string Validation::Key(const Json& issue) {
    return jst::fmt::format("{}|{}|{}",
                            issue.value("code", std::string{}),
                            issue.value("block", std::string{}),
                            issue.value("input", std::string{}));
}

Json Validation::Health(const Json& issues, U64 total) {
    return Json({
        {"valid", total == 0},
        {"issue_count", total},
        {"issues", issues},
    });
}

}  // namespace Jetstream::Agent::Detail
