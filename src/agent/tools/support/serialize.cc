#include "base.hh"
#include "jetstream/flowgraph_metadata.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "tools/text.hh"

namespace Jetstream::Agent::Detail {

nlohmann::json Serialize::Implementations(const std::string& type) {
    nlohmann::json implementations = nlohmann::json::array();

    const auto blocks = Registry::ListAvailableBlocks(type);
    const bool composite = !blocks.empty() && blocks.front().moduleRequirements.size() != 1;
    for (const auto& target : Registry::ListAvailableBlockTargets(type)) {
        implementations.push_back({
            {"device", GetDeviceName(target.device)},
            {"runtime", GetRuntimeName(target.runtime)},
            {"provider", target.provider},
            {"kind", composite ? "composite" : "direct"},
        });
    }

    return implementations;
}

namespace {

nlohmann::json CompactJson(const nlohmann::json& value) {
    if (value.is_string()) {
        const auto text = value.get<std::string>();
        if (text.size() > 256) {
            return jst::fmt::format("{}... ({} bytes, inspect_block shows all)",
                                    Text::Utf8Prefix(text, 256), text.size());
        }
        return text;
    }

    if (value.is_object()) {
        nlohmann::json result = nlohmann::json::object();
        for (auto it = value.begin(); it != value.end(); ++it) {
            result[it.key()] = CompactJson(it.value());
        }
        return result;
    }

    if (value.is_array()) {
        nlohmann::json result = nlohmann::json::array();

        const U64 shown = std::min<U64>(value.size(), 16);
        for (U64 i = 0; i < shown; ++i) {
            result.push_back(CompactJson(value[i]));
        }

        if (shown < value.size()) {
            result.push_back(jst::fmt::format("... ({} items, inspect_block shows all)", value.size()));
        }
        return result;
    }

    return value;
}

nlohmann::json DropdownOptionsJson(const Parser::Map& format) {
    nlohmann::json json;
    if (!Value::TryToJson(format, json)) {
        return nullptr;
    }

    if (!json.is_object() || json.value("type", std::string{}) != "dropdown") {
        return nullptr;
    }

    if (!json.contains("options") || !json["options"].is_array()) {
        return nullptr;
    }

    nlohmann::json options = nlohmann::json::array();
    for (const auto& option : json["options"]) {
        if (!option.is_object() || !option.contains("value")) {
            continue;
        }

        nlohmann::json item = {{"value", option["value"]}};
        if (option.contains("label") && option["label"] != option["value"]) {
            item["label"] = option["label"];
        }
        options.push_back(std::move(item));
    }

    return options;
}

}  // namespace

Result Serialize::Collect(const std::shared_ptr<Flowgraph>& flowgraph,
                          std::vector<Flowgraph::View::BlockData>& blocks) {
    std::vector<std::string> names;
    JST_CHECK(flowgraph->view().keys(names));

    blocks.clear();
    blocks.reserve(names.size());
    for (const auto& name : names) {
        Flowgraph::View::BlockData data;
        if (flowgraph->view().block(name, data) == Result::SUCCESS) {
            blocks.push_back(std::move(data));
        }
    }

    return Result::SUCCESS;
}

bool Serialize::HasPort(const std::vector<Flowgraph::View::InterfaceEntry>& entries, const std::string& port) {
    return std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
        return entry.name == port;
    });
}

nlohmann::json Serialize::PortNames(const std::vector<Flowgraph::View::InterfaceEntry>& entries) {
    nlohmann::json names = nlohmann::json::array();
    for (const auto& entry : entries) {
        names.push_back(entry.name);
    }
    return names;
}

nlohmann::json Serialize::Port(const std::string& block, const std::string& port) {
    return {
        {"block", block},
        {"port", port},
    };
}

nlohmann::json Serialize::Connection(const std::string& sourceBlock,
                                     const std::string& sourcePort,
                                     const std::string& destinationBlock,
                                     const std::string& destinationPort) {
    return {
        {"source", Serialize::Port(sourceBlock, sourcePort)},
        {"destination", Serialize::Port(destinationBlock, destinationPort)},
    };
}

nlohmann::json Serialize::Inputs(const Flowgraph::View::BlockData& data) {
    nlohmann::json inputs = nlohmann::json::object();
    for (const auto& [port, link] : data.inputs) {
        if (link.external.has_value()) {
            inputs[port] = Serialize::Port(link.external->block, link.external->port);
        }
    }
    return inputs;
}

std::optional<nlohmann::json> Serialize::PendingConfigOptions(const Flowgraph::View::BlockData& data) {
    if (data.state != Block::State::Errored && data.state != Block::State::Incomplete) {
        return std::nullopt;
    }
    return Serialize::ConfigOptions(data.interfaceConfigs);
}

nlohmann::json Serialize::Map(const Parser::Map& map) {
    nlohmann::json json;
    if (Value::ToJson(map, json) != Result::SUCCESS) {
        return nullptr;
    }
    return json;
}

nlohmann::json Serialize::CompactMap(const Parser::Map& map) {
    return CompactJson(Serialize::Map(map));
}

nlohmann::json Serialize::Interface(const std::vector<Flowgraph::View::InterfaceEntry>& entries, bool help) {
    nlohmann::json result = nlohmann::json::array();

    for (const auto& entry : entries) {
        nlohmann::json item = {{"name", entry.name}};

        if (!entry.label.empty()) {
            item["label"] = entry.label;
        }

        if (help && !entry.help.empty()) {
            item["help"] = entry.help;
        }

        if (auto options = DropdownOptionsJson(entry.format); !options.is_null()) {
            item["options"] = std::move(options);
        }

        result.push_back(std::move(item));
    }

    return result;
}

nlohmann::json Serialize::ConfigOptions(const std::vector<Flowgraph::View::InterfaceEntry>& entries) {
    nlohmann::json result = nlohmann::json::object();
    for (const auto& entry : entries) {
        if (auto options = DropdownOptionsJson(entry.format); !options.is_null()) {
            result[entry.name] = std::move(options);
        }
    }
    return result;
}

nlohmann::json Serialize::Node(const std::shared_ptr<Flowgraph>& flowgraph, const std::string& name) {
    Flowgraph::Metadata::NodeLayout layout;
    flowgraph->metadata().get("node", layout, name);

    nlohmann::json node = {
        {"x", layout.x},
        {"y", layout.y},
    };

    if (layout.width != 0.0f) {
        node["width"] = layout.width;
    }

    if (layout.height != 0.0f) {
        node["height"] = layout.height;
    }

    return node;
}

nlohmann::json Serialize::NodeLayout(const std::shared_ptr<Flowgraph>& flowgraph, const std::string& name) {
    return {
        {"node", Serialize::Node(flowgraph, name)},
        {"layout_status", "pending"},
        {"bounds", nullptr},
    };
}

nlohmann::json Serialize::BlockDetail(const std::shared_ptr<Flowgraph>& flowgraph,
                                      const Flowgraph::View::BlockData& data,
                                      bool full) {
    nlohmann::json entry = {
        {"name", data.name},
        {"type", data.type},
        {"title", data.title},
    };

    if (full || data.device != DeviceType::CPU) {
        entry["device"] = GetDeviceName(data.device);
    }

    if (full || data.runtime != RuntimeType::NATIVE) {
        entry["runtime"] = GetRuntimeName(data.runtime);
    }

    if (full || data.provider != "generic") {
        entry["provider"] = data.provider;
    }

    entry["state"] = GetBlockStateName(data.state);

    if (!data.diagnostic.empty()) {
        entry["diagnostic"] = data.diagnostic;
    }

    entry["config"] = full ? Serialize::Map(data.config) : Serialize::CompactMap(data.config);
    entry["inputs"] = Serialize::Inputs(data);

    if (full) {
        entry["summary"] = data.summary;
        entry["config_fields"] = Serialize::Interface(data.interfaceConfigs, false);
        entry["input_ports"] = Serialize::Interface(data.interfaceInputs, false);
        entry["output_ports"] = Serialize::Interface(data.interfaceOutputs, false);
    } else {
        entry["input_ports"] = Serialize::PortNames(data.interfaceInputs);
        entry["output_ports"] = Serialize::PortNames(data.interfaceOutputs);
    }

    entry.update(Serialize::NodeLayout(flowgraph, data.name));
    return entry;
}

nlohmann::json Serialize::FlowgraphDetail(const std::shared_ptr<Flowgraph>& flowgraph, bool full) {
    std::vector<Flowgraph::View::BlockData> entries;
    if (Serialize::Collect(flowgraph, entries) != Result::SUCCESS) {
        return {{"error", "Could not inspect the selected flowgraph."}};
    }

    std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
        return a.name < b.name;
    });

    nlohmann::json blocks = nlohmann::json::array();
    for (const auto& data : entries) {
        blocks.push_back(Serialize::BlockDetail(flowgraph, data, full));
    }

    return {
        {"block_count", blocks.size()},
        {"blocks", std::move(blocks)},
    };
}

}  // namespace Jetstream::Agent::Detail
