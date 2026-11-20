#include "flowgraphs.hh"

#include "../../model/callbacks.hh"
#include "../../model/state.hh"

#include <nlohmann/json.hpp>

#include <algorithm>

namespace Jetstream {

namespace {

using Json = nlohmann::json;

constexpr const char* kList = "list_flowgraphs";
constexpr const char* kOpen = "open_flowgraph";

Agent::Tools::Output Reply(const Json& body) {
    return {body.dump(-1, ' ', false, Json::error_handler_t::replace)};
}

bool Listed(const std::vector<Settings::Interface::FlowgraphEntry>& entries, const std::string& path) {
    return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.path == path;
    });
}

}  // namespace

AgentFlowgraphs::AgentFlowgraphs(DefaultCompositorState& state, const DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

bool AgentFlowgraphs::Handles(const std::string& name) {
    return name == kList || name == kOpen;
}

std::vector<Agent::Tools::Definition> AgentFlowgraphs::Definitions() {
    return {
        {
            kList,
            "List the user's favorite and recently opened flowgraph files, the same ones the Start tab shows. "
            "Returns each file's path and title, with favorite and recent flags. Bundled examples are separate "
            "and come from list_examples.",
            R"({"type":"object","properties":{},"additionalProperties":false})",
            {.permission = Agent::Tools::Permission::Read},
        },
        {
            kOpen,
            "Open one of the user's favorite or recent flowgraph files as a new graph, the same way the Start "
            "tab opens it, and select the new graph for subsequent calls. Graphs that are already open stay "
            "open and unchanged. Returns the opened path.",
            R"({"type":"object","required":["path"],"properties":{"path":{"type":"string","description":"Exact path from list_flowgraphs."}},"additionalProperties":false})",
        },
    };
}

void AgentFlowgraphs::execute(const std::string& name, const std::string& args, Completion done) const {
    if (name == kList) {
        done(list(), nullptr);
        return;
    }
    open(args, std::move(done));
}

Agent::Tools::Output AgentFlowgraphs::list() const {
    const auto& favorites = state.interface.favoriteFlowgraphs;
    const auto& recent = state.interface.recentFlowgraphs;

    Json flowgraphs = Json::array();
    const auto add = [&](const Settings::Interface::FlowgraphEntry& entry) {
        Json item = {
            {"path", entry.path},
            {"favorite", Listed(favorites, entry.path)},
            {"recent", Listed(recent, entry.path)},
        };
        if (!entry.title.empty()) {
            item["title"] = entry.title;
        }
        flowgraphs.push_back(std::move(item));
    };

    for (const auto& entry : favorites) {
        add(entry);
    }
    for (const auto& entry : recent) {
        if (!Listed(favorites, entry.path)) {
            add(entry);
        }
    }

    return Reply({{"flowgraphs", std::move(flowgraphs)}});
}

void AgentFlowgraphs::open(const std::string& args, Completion done) const {
    const auto request = Json::parse(args, nullptr, false);
    if (!request.is_object()) {
        done(Agent::Tools::Error("Arguments must be a JSON object."), nullptr);
        return;
    }
    for (const auto& [key, _] : request.items()) {
        if (key != "path") {
            done(Agent::Tools::Error(jst::fmt::format("Unknown field '{}'.", key)), nullptr);
            return;
        }
    }
    if (!request.contains("path") || !request["path"].is_string()) {
        done(Agent::Tools::Error("open_flowgraph requires a string 'path' from list_flowgraphs."), nullptr);
        return;
    }

    const auto path = request["path"].get<std::string>();
    const auto& interface = state.interface;
    if (!Listed(interface.favoriteFlowgraphs, path) && !Listed(interface.recentFlowgraphs, path)) {
        done(Agent::Tools::Error(jst::fmt::format("Flowgraph '{}' is not a favorite or recent flowgraph. "
                                                   "Use an exact path from list_flowgraphs.", path)), nullptr);
        return;
    }

    callbacks.enqueueMail(MailOpenFlowgraphPath{
        .path = path,
        .onComplete = [path, done = std::move(done)](Result result, std::shared_ptr<Flowgraph> flowgraph) {
            if (result == Result::SUCCESS) {
                done(Reply({{"opened", path}}), std::move(flowgraph));
            } else if (flowgraph) {
                done(Agent::Tools::Error(jst::fmt::format("Could not load flowgraph '{}'. An empty graph was "
                                                           "opened in its place.", path)), nullptr);
            } else {
                done(Agent::Tools::Error(jst::fmt::format("Could not open flowgraph '{}'.", path)), nullptr);
            }
        },
    });
}

}  // namespace Jetstream
