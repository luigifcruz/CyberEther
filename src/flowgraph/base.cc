#include <regex>

#include "jetstream/flowgraph.hh"
#include "jetstream/instance.hh"
#include "jetstream/store.hh"

#include "yaml.hh"

namespace Jetstream {

Flowgraph::Flowgraph(Instance& instance) : _instance(instance) {
    _yaml = std::make_unique<YamlImpl>();
}

Flowgraph::~Flowgraph() {
    _yaml.reset();
}

Result Flowgraph::create() {
    JST_INFO("[FLOWGRAPH] Creating flowgraph in-memory.");

    if (_created) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is already created.");
        return Result::ERROR;
    }

    _protocolVersion = "1.0.0";
    _cyberetherVersion = JETSTREAM_VERSION_STR;
    _created = true;

    return Result::SUCCESS;
}

Result Flowgraph::destroy() {
    JST_DEBUG("[FLOWGRAPH] Destroying flowgraph.");

    if (!_created) {
        return Result::SUCCESS;
    }

    _protocolVersion.clear();
    _cyberetherVersion.clear();
    _title.clear();
    _summary.clear();
    _author.clear();
    _license.clear();
    _description.clear();
    _yaml->data.clear();

    _nodes.clear();
    _nodesOrder.clear();

    _created = false;

    return Result::SUCCESS;
}

Result Flowgraph::exportToFile(const std::string& path) {
    if (path.empty()) {
        JST_ERROR("[FLOWGRAPH] Filepath is empty.");
        return Result::ERROR;
    }

    JST_INFO("[FLOWGRAPH] Exporting flowgraph to file '{}'.", path);

    std::vector<char> blob;
    JST_CHECK(exportToBlob(blob));

    std::fstream file(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
        JST_ERROR("[FLOWGRAPH] Can't open flowgraph file '{}'.", path);
        return Result::ERROR;
    }
    file.write("---\n", 4);
    file.write(blob.data(), blob.size());
    file.close();

    return Result::SUCCESS;
}

Result Flowgraph::exportToBlob(std::vector<char>& blob) {
    JST_DEBUG("[FLOWGRAPH] Exporting flowgraph to blob.");

    _yaml->data = ryml::Tree();
    ryml::NodeRef root = _yaml->data.rootref();
    root |= ryml::MAP;

    root["protocolVersion"] << _protocolVersion;
    root["cyberetherVersion"] << _cyberetherVersion;

    if (!_title.empty()) {
        root["title"] << _title;
    }

    if (!_summary.empty()) {
        root["summary"] << _summary;
    }

    if (!_author.empty()) {
        root["author"] << _author;
    }

    if (!_license.empty()) {
        root["license"] << _license;
    }

    if (!_description.empty()) {
        root["description"] << _description;
        if (std::count(_description.begin(), _description.end(), '\n')) {
            root["description"] |= ryml::_WIP_VAL_LITERAL;
        }
    }

    ryml::NodeRef graph;
    if (!_nodesOrder.empty()) {
        graph = root["graph"];
        graph |= ryml::MAP;
    }

    for (auto& locale : _nodesOrder) {
        // Fetch node from the map.
        auto& node = _nodes.at(locale);

        // Ignore modules.

        if (!node->block) {
            continue;
        }

        // Update maps.
        JST_CHECK(node->updateMaps());

        // List node metadata.

        auto& fingerprint = node->fingerprint;
        auto& configMap = node->configMap;
        auto& inputMap = node->inputMap;
        auto& stateMap = node->stateMap;

        // TODO: Ignore default values.

        // Create block fingerprint.

        ryml::NodeRef block = graph[ryml::to_csubstr(locale.blockId)];
        block |= ryml::MAP;

        block["module"] << fingerprint.id;
        block["device"] << fingerprint.device;

        if (!fingerprint.inputDataType.empty() &&
            fingerprint.outputDataType.empty()) {
            block["dataType"] << fingerprint.inputDataType;
        } else if (!fingerprint.inputDataType.empty() &&
                   !fingerprint.outputDataType.empty()) {
            block["inputDataType"] << fingerprint.inputDataType;
            block["outputDataType"] << fingerprint.outputDataType;
        }

        // Create block configuration.

        if (!configMap.empty()) {
            ryml::NodeRef config = block["config"];
            config |= ryml::MAP;

            for (const auto& [key, value] : configMap) {
                std::string string_value;
                JST_CHECK(Parser::SerializeToString(value, string_value));

                if (string_value.empty()) {
                    continue;
                }

                const auto& k = ryml::to_csubstr(key);
                config[k] << string_value;
                if (std::count(string_value.begin(), string_value.end(), '\n')) {
                    config[k] |= ryml::_WIP_VAL_LITERAL;
                } else {
                    config[k] |= ryml::_WIP_VAL_PLAIN;
                }
            }
        }

        // Create block input.

        if (!inputMap.empty()) {
            ryml::NodeRef input = block["input"];
            input |= ryml::MAP;

            for (const auto& [key, value] : inputMap) {
                std::string string_value;
                JST_CHECK(Parser::SerializeToString(value, string_value));

                if (string_value.empty()) {
                    continue;
                }

                const auto& k = ryml::to_csubstr(key);
                input[k] << string_value;
                input[k] |= ryml::_WIP_VAL_PLAIN;
            }
        }

        // Create block interface.

        if (!stateMap.empty()) {
            ryml::NodeRef interface = block["interface"];
            interface |= ryml::MAP;

            for (const auto& [key, value] : stateMap) {
                std::string string_value;
                JST_CHECK(Parser::SerializeToString(value, string_value));

                if (string_value.empty()) {
                    continue;
                }

                const auto& k = ryml::to_csubstr(key);
                interface[k] << string_value;
                interface[k] |= ryml::_WIP_VAL_PLAIN;
            }
        }
    }

    // Emit YAML to blob.
    blob = ryml::emitrs_yaml<std::vector<char>>(_yaml->data);

    return Result::SUCCESS;
}

Result Flowgraph::importFromFile(const std::string& path) {
    std::fstream file;

    if (!std::filesystem::exists(path)) {
        JST_INFO("[FLOWGRAPH] Creating flowgraph file '{}'.", path);
        file = std::fstream(path, std::ios::in | std::ios::binary);
    } else {
        JST_INFO("[FLOWGRAPH] Opening flowgraph file '{}'.", path);
        file = std::fstream(path, std::ios::in | std::ios::binary);
    }

    if (!file) {
        JST_ERROR("[FLOWGRAPH] Can't open flowgraph file '{}'.", path);
        return Result::ERROR;
    }

    const auto filesize = std::filesystem::file_size(path);
    std::vector<char> blob(filesize, 0);
    file.read(blob.data(), filesize);
    file.close();

    JST_CHECK(importFromBlob(blob));

    return Result::SUCCESS;
}

Result Flowgraph::importFromBlob(const std::vector<char>& blob) {
    _yaml->data = ryml::parse({blob.data(), blob.size()});

    if (_yaml->data.rootref().empty()) {
        return Result::SUCCESS;
    }

    auto root = _yaml->data.rootref()[0];
    auto configValues = YamlImpl::GatherNodes(root, root, {"protocolVersion",
                                                           "cyberetherVersion"});

    _protocolVersion = YamlImpl::ResolveReadable(configValues["protocolVersion"]);
    _cyberetherVersion = YamlImpl::ResolveReadable(configValues["cyberetherVersion"]);

    if (_protocolVersion != "1.0.0") {
        JST_ERROR("[FLOWGRAPH] Invalid protocol version '{}'.", _protocolVersion);
        return Result::ERROR;
    }

    if (_cyberetherVersion != JETSTREAM_VERSION_STR) {
        JST_WARN("[FLOWGRAPH] Flowgraph was created with a different version of CyberEther ({}).", _cyberetherVersion);
    }

    auto optConfigValues = YamlImpl::GatherNodes(root, root, {"title",
                                                              "summary",
                                                              "author",
                                                              "license",
                                                              "description"}, true);

    if (optConfigValues.contains("title")) {
        _title = YamlImpl::ResolveReadable(optConfigValues["title"]);
    }
    if (optConfigValues.contains("summary")) {
        _summary = YamlImpl::ResolveReadable(optConfigValues["summary"]);
    }
    if (optConfigValues.contains("author")) {
        _author = YamlImpl::ResolveReadable(optConfigValues["author"]);
    }
    if (optConfigValues.contains("license")) {
        _license = YamlImpl::ResolveReadable(optConfigValues["license"]);
    }
    if (optConfigValues.contains("description")) {
        _description = YamlImpl::ResolveReadable(optConfigValues["description"]);
    }

    if (!YamlImpl::HasNode(root, root, "graph")) {
        return Result::SUCCESS;
    }
    
    for (const auto& node : YamlImpl::GetNode(root, root, "graph")) {
        const auto nodeKey = YamlImpl::ResolveReadableKey(node);
        JST_DEBUG("[FLOWGRAPH] Processing '{}' module.", nodeKey);

        Block::Fingerprint fingerprint;
        Parser::RecordMap inputMap;
        Parser::RecordMap configMap;
        Parser::RecordMap stateMap;

        // Populate fingerprint.

        auto values = YamlImpl::GatherNodes(root, node, {"module", "device", "dataType", "inputDataType", "outputDataType"}, true);
        fingerprint.id = YamlImpl::ResolveReadable(values["module"]);
        fingerprint.device = YamlImpl::ResolveReadable(values["device"]);
        if (values.contains("dataType")) {
            fingerprint.inputDataType = YamlImpl::ResolveReadable(values["dataType"]);
        } else if (values.contains("inputDataType") && values.contains("outputDataType")) {
            fingerprint.inputDataType = YamlImpl::ResolveReadable(values["inputDataType"]);
            fingerprint.outputDataType = YamlImpl::ResolveReadable(values["outputDataType"]);
        }

        // Populate data.

        if (YamlImpl::HasNode(root, node, "config")) {
            for (const auto& element : YamlImpl::GetNode(root, node, "config")) {
                auto localPlaceholder = YamlImpl::SolvePlaceholder(root, element);
                configMap[YamlImpl::ResolveReadableKey(element)] = _yaml->solveLocalPlaceholder(_nodes, localPlaceholder);
            }
        }

        if (YamlImpl::HasNode(root, node, "input")) {
            for (const auto& element : YamlImpl::GetNode(root, node, "input")) {
                auto localPlaceholder = YamlImpl::SolvePlaceholder(root, element);
                inputMap[YamlImpl::ResolveReadableKey(element)] = _yaml->solveLocalPlaceholder(_nodes, localPlaceholder);
            }
        }

        if (YamlImpl::HasNode(root, node, "interface")) {
            for (const auto& element : YamlImpl::GetNode(root, node, "interface")) {
                auto localPlaceholder = YamlImpl::SolvePlaceholder(root, element);
                stateMap[YamlImpl::ResolveReadableKey(element)] = _yaml->solveLocalPlaceholder(_nodes, localPlaceholder);
            }
        }

        if (!Store::BlockConstructorList().contains(fingerprint)) {
            JST_ERROR("[FLOWGRAPH] Can't find module with such a signature ({}).", fingerprint);
            return Result::ERROR;
        }

        JST_CHECK(Store::BlockConstructorList().at(fingerprint)(_instance, nodeKey, configMap, inputMap, stateMap));
    }

    return Result::SUCCESS;
}

Result Flowgraph::setTitle(const std::string& title) {
    if (!_created) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not create.");
        return Result::ERROR;
    }

    _title = title;

    return Result::SUCCESS;
}

Result Flowgraph::setSummary(const std::string& summary) {
    if (!_created) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not create.");
        return Result::ERROR;
    }

    _summary = summary;

    return Result::SUCCESS;
}

Result Flowgraph::setAuthor(const std::string& author) {
    if (!_created) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not create.");
        return Result::ERROR;
    }

    _author = author;

    return Result::SUCCESS;
}

Result Flowgraph::setLicense(const std::string& license) {
    if (!_created) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not create.");
        return Result::ERROR;
    }

    _license = license;

    return Result::SUCCESS;
}

Result Flowgraph::setDescription(const std::string& description) {
    if (!_created) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not create.");
        return Result::ERROR;
    }

    _description = description;

    return Result::SUCCESS;
}

Result Flowgraph::print() const {
    if (!_created) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not create.");
        return Result::ERROR;
    }

    if (_yaml->data.rootref().empty()) {
        return Result::SUCCESS;
    }

    auto root = _yaml->data.rootref()[0];
    auto configValues = YamlImpl::GatherNodes(root, root, {"protocolVersion",
                                                           "cyberetherVersion"});

    auto optConfigValues = YamlImpl::GatherNodes(root, root, {"title",
                                                              "summary",
                                                              "author",
                                                              "license",
                                                              "description"}, true);

    JST_INFO("-----------------------------------------------------------");
    JST_INFO("|                  JETSTREAM CONFIG FILE                  |")
    JST_INFO("-----------------------------------------------------------");
    JST_INFO("Protocol Version:   {}", YamlImpl::ResolveReadable(configValues["protocolVersion"]));
    JST_INFO("CyberEther Version: {}", YamlImpl::ResolveReadable(configValues["cyberetherVersion"]));
    if (optConfigValues.contains("title")) {
        JST_INFO("Title:              {}", YamlImpl::ResolveReadable(optConfigValues["title"]));
    }
    if (optConfigValues.contains("summary")) {
        JST_INFO("Summary:        {}", YamlImpl::ResolveReadable(optConfigValues["summary"]));
    }
    if (optConfigValues.contains("author")) {
        JST_INFO("Author:            {}", YamlImpl::ResolveReadable(optConfigValues["author"]));
    }
    if (optConfigValues.contains("license")) {
        JST_INFO("License:            {}", YamlImpl::ResolveReadable(optConfigValues["license"]));
    }
    if (optConfigValues.contains("description")) {
        JST_INFO("Description:        {}", YamlImpl::ResolveReadable(optConfigValues["description"]));
    }

    if (!YamlImpl::HasNode(root, root, "graph")) {
        return Result::SUCCESS;
    }

    JST_INFO("------------------------- GRAPH --------------------------");

    for (const auto& node : YamlImpl::GetNode(root, root, "graph")) {
        auto values = YamlImpl::GatherNodes(root, node, {"module",
                                                         "device",
                                                         "dataType",
                                                         "inputDataType",
                                                         "outputDataType"}, true);

        JST_INFO("[{}]:", YamlImpl::ResolveReadableKey(node));
        JST_INFO("  Module:            {}", YamlImpl::ResolveReadable(values["module"]));
        JST_INFO("  Device:            {}", YamlImpl::ResolveReadable(values["device"]));

        if (values.contains("dataType")) {
            JST_INFO("  Data Type:         {}", YamlImpl::ResolveReadable(values["dataType"]));
        } else if (values.contains("inputDataType") && values.contains("outputDataType")) {
            JST_INFO("  Data Type:         {} -> {}", YamlImpl::ResolveReadable(values["inputDataType"]),
                                                      YamlImpl::ResolveReadable(values["outputDataType"]));
        }

        if (YamlImpl::HasNode(root, node, "config")) {
            JST_INFO("  Config:");
            for (const auto& element : YamlImpl::GetNode(root, node, "config")) {
                JST_INFO("    {} = {}", YamlImpl::ResolveReadableKey(element), YamlImpl::ResolveReadable(YamlImpl::SolvePlaceholder(root, element)));
            }
        }

        if (YamlImpl::HasNode(root, node, "input")) {
            JST_INFO("  Input:");
            for (const auto& element : YamlImpl::GetNode(root, node, "input")) {
                JST_INFO("    {} = {}", YamlImpl::ResolveReadableKey(element), YamlImpl::ResolveReadable(YamlImpl::SolvePlaceholder(root, element)));
            }
        }

        if (YamlImpl::HasNode(root, node, "interface")) {
            JST_INFO("  Interface:");
            for (const auto& element : YamlImpl::GetNode(root, node, "interface")) {
                JST_INFO("    {} = {}", YamlImpl::ResolveReadableKey(element), YamlImpl::ResolveReadable(YamlImpl::SolvePlaceholder(root, element)));
            }
        }
    }

    JST_INFO("-----------------------------------------------------------");

    return Result::SUCCESS;
}

}  // namespace Jetstream
