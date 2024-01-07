#include <regex>

#include "jetstream/flowgraph.hh"
#include "jetstream/instance.hh"
#include "jetstream/store.hh"

namespace Jetstream {

Flowgraph::Flowgraph(Instance& instance) : _instance(instance) {}

Result Flowgraph::create() {
    JST_INFO("[FLOWGRAPH] Creating flowgraph in-memory.");

    if (_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is already imported.");
        return Result::ERROR;
    }

    _protocolVersion = "1.0.0";
    _cyberetherVersion = JETSTREAM_VERSION_STR;

    JST_CHECK(importFromBlob());

    return Result::SUCCESS;
}

Result Flowgraph::create(const std::string& path) {
    JST_CHECK(setFilename(path));

    std::fstream file;

    if (!std::filesystem::exists(_filename)) {
        JST_INFO("[FLOWGRAPH] Creating flowgraph file '{}'.", _filename);
        file = std::fstream(_filename, std::ios::in | std::ios::binary);
    } else {
        JST_INFO("[FLOWGRAPH] Opening flowgraph file '{}'.", _filename);
        file = std::fstream(_filename, std::ios::in | std::ios::binary);
    }

    if (!file) {
        JST_ERROR("[FLOWGRAPH] Can't open flowgraph file '{}'.", _filename);
        return Result::ERROR;
    }

    const auto filesize = std::filesystem::file_size(_filename);
    _blob.resize(filesize);
    file.read(_blob.data(), filesize);
    file.close();

    _protocolVersion = "1.0.0";
    _cyberetherVersion = JETSTREAM_VERSION_STR;

    JST_CHECK(importFromBlob());

    return Result::SUCCESS;
}

Result Flowgraph::create(const char* blob) {
    JST_INFO("[FLOWGRAPH] Creating flowgraph in-memory from blob.");

    if (_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is already imported.");
        return Result::ERROR;
    }

    if (blob == nullptr) {
        JST_ERROR("[FLOWGRAPH] Blob is empty.");
        return Result::ERROR;
    }

    _blob.insert(_blob.begin(), blob, blob + strlen(blob));
    
    // Add a newline at the end of the file in case of an empty file.
    if (_blob.empty() || _blob.back() != '\n') {
        _blob.push_back('\n');
    }

    _protocolVersion = "1.0.0";
    _cyberetherVersion = JETSTREAM_VERSION_STR;

    JST_CHECK(importFromBlob());

    return Result::SUCCESS;
}

Result Flowgraph::destroy() {
    JST_DEBUG("[FLOWGRAPH] Destroying flowgraph.");

    if (!_imported) {
        return Result::SUCCESS;
    }

    _protocolVersion.clear();
    _cyberetherVersion.clear();
    _title.clear();
    _summary.clear();
    _author.clear();
    _license.clear();
    _description.clear();
    _yaml.clear();

    _filename.clear();
    _blob.clear();
    _nodes.clear();
    _nodesOrder.clear();

    _imported = false;

    return Result::SUCCESS;
}

Result Flowgraph::exportToFile() {
    if (_filename.empty()) {
        JST_ERROR("[FLOWGRAPH] Filepath is empty.");
        return Result::ERROR;
    }

    JST_INFO("[FLOWGRAPH] Exporting flowgraph to file '{}'.", _filename);

    if (!_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not imported.");
        return Result::ERROR;
    }

    JST_CHECK(exportToBlob());

    std::fstream file(_filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!file) {
        JST_ERROR("[FLOWGRAPH] Can't open flowgraph file '{}'.", _filename);
        return Result::ERROR;
    }
    file.write("---\n", 4);
    file.write(_blob.data(), _blob.size());
    file.close();

    return Result::SUCCESS;
}

Result Flowgraph::exportToBlob() {
    if (!_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not imported.");
        return Result::ERROR;
    }

    JST_DEBUG("[FLOWGRAPH] Exporting flowgraph to blob.");

    _yaml = ryml::Tree();
    ryml::NodeRef root = _yaml.rootref();
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
    _blob = ryml::emitrs_yaml<std::vector<char>>(_yaml);

    return Result::SUCCESS;
}

Result Flowgraph::importFromBlob() {
    if (_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is already imported.");
        return Result::ERROR;
    }

    _yaml = ryml::parse_in_place({_blob.data(), _blob.size()});
    _imported = true;

    if (_yaml.rootref().empty()) {
        return Result::SUCCESS;
    }

    auto root = _yaml.rootref()[0];
    auto configValues = GatherNodes(root, root, {"protocolVersion",
                                                 "cyberetherVersion"});

    _protocolVersion = ResolveReadable(configValues["protocolVersion"]);
    _cyberetherVersion = ResolveReadable(configValues["cyberetherVersion"]);

    if (_protocolVersion != "1.0.0") {
        JST_ERROR("[FLOWGRAPH] Invalid protocol version '{}'.", _protocolVersion);
        return Result::ERROR;
    }

    if (_cyberetherVersion != JETSTREAM_VERSION_STR) {
        JST_WARN("[FLOWGRAPH] Flowgraph was created with a different version of CyberEther ({}).", _cyberetherVersion);
    }

    auto optConfigValues = GatherNodes(root, root, {"title",
                                                    "summary",
                                                    "author",
                                                    "license",
                                                    "description"}, true);

    if (optConfigValues.contains("title")) {
        _title = ResolveReadable(optConfigValues["title"]);
    }
    if (optConfigValues.contains("summary")) {
        _summary = ResolveReadable(optConfigValues["summary"]);
    }
    if (optConfigValues.contains("author")) {
        _author = ResolveReadable(optConfigValues["author"]);
    }
    if (optConfigValues.contains("license")) {
        _license = ResolveReadable(optConfigValues["license"]);
    }
    if (optConfigValues.contains("description")) {
        _description = ResolveReadable(optConfigValues["description"]);
    }

    if (!HasNode(root, root, "graph")) {
        return Result::SUCCESS;
    }
    
    for (const auto& node : GetNode(root, root, "graph")) {
        const auto nodeKey = ResolveReadableKey(node);
        JST_DEBUG("[FLOWGRAPH] Processing '{}' module.", nodeKey);

        Block::Fingerprint fingerprint;
        Parser::RecordMap inputMap;
        Parser::RecordMap configMap;
        Parser::RecordMap stateMap;

        // Populate fingerprint.

        auto values = GatherNodes(root, node, {"module", "device", "dataType", "inputDataType", "outputDataType"}, true);
        fingerprint.id = ResolveReadable(values["module"]);
        fingerprint.device = ResolveReadable(values["device"]);
        if (values.contains("dataType")) {
            fingerprint.inputDataType = ResolveReadable(values["dataType"]);
        } else if (values.contains("inputDataType") && values.contains("outputDataType")) {
            fingerprint.inputDataType = ResolveReadable(values["inputDataType"]);
            fingerprint.outputDataType = ResolveReadable(values["outputDataType"]);
        }

        // Populate data.

        if (HasNode(root, node, "config")) {
            for (const auto& element : GetNode(root, node, "config")) {
                auto localPlaceholder = SolvePlaceholder(root, element);
                configMap[ResolveReadableKey(element)] = solveLocalPlaceholder(localPlaceholder);
            }
        }

        if (HasNode(root, node, "input")) {
            for (const auto& element : GetNode(root, node, "input")) {
                auto localPlaceholder = SolvePlaceholder(root, element);
                inputMap[ResolveReadableKey(element)] = solveLocalPlaceholder(localPlaceholder);
            }
        }

        if (HasNode(root, node, "interface")) {
            for (const auto& element : GetNode(root, node, "interface")) {
                auto localPlaceholder = SolvePlaceholder(root, element);
                stateMap[ResolveReadableKey(element)] = solveLocalPlaceholder(localPlaceholder);
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

Parser::Record Flowgraph::solveLocalPlaceholder(const ryml::ConstNodeRef& node) {
    if (!node.has_val()) {
        return {ResolveReadable(node)};
    }

    std::string key = std::string(node.val().str, node.val().len);

    std::regex placeholderPattern(R"(\$\{.*\})");
    if (!std::regex_match(key, placeholderPattern)) {
        return {ResolveReadable(node)};
    }

    std::vector<std::string> patternNodes = GetParameterNodes(GetParameterContents(key));

    if (patternNodes.size() != 4) {
        JST_ERROR("[PARSER] Variable '{}' not found.", key);
        JST_CHECK_THROW(Result::ERROR);
    }

    const auto& graphKey = patternNodes[0];
    const auto& moduleKey = patternNodes[1];
    const auto& arrayKey = patternNodes[2];
    const auto& elementKey = patternNodes[3];

    if (graphKey.compare("graph")) {
        JST_ERROR("[PARSER] Invalid variable '{}'.", key);
        JST_CHECK_THROW(Result::ERROR);
    }

    if (!_nodes.contains({"", moduleKey})) {
        JST_ERROR("[PARSER] Module from the variable '{}' not found.", key);
        JST_CHECK_THROW(Result::ERROR);
    }

    if (!arrayKey.compare("input")) {
        auto& map = _nodes.at({"", moduleKey})->inputMap;
        if (!map.contains(elementKey)) {
            JST_ERROR("[PARSER] Input element from the variable '{}' not found.", key);
            JST_CHECK_THROW(Result::ERROR);
        }
        return map[elementKey];
    }

    if (!arrayKey.compare("output")) {
        auto& map = _nodes.at({"", moduleKey})->outputMap;
        if (!map.contains(elementKey)) {
            JST_ERROR("[PARSER] Output element from the variable '{}' not found.", key);
            JST_CHECK_THROW(Result::ERROR);
        }
        return map[elementKey];
    }

    if (!arrayKey.compare("interface")) {
        auto& map = _nodes.at({"", moduleKey})->stateMap;
        if (!map.contains(elementKey)) {
            JST_ERROR("[PARSER] Interface state element from the variable '{}' not found.", key);
            JST_CHECK_THROW(Result::ERROR);
        }
        return map[elementKey];
    }

    JST_ERROR("[PARSER] Invalid module array '{}'. It should be 'input', 'output', or 'interface'.", key);
    throw Result::ERROR;
}

Result Flowgraph::setFilename(const std::string& filename) {
    if (filename.empty()) {
        JST_ERROR("[FLOWGRAPH] Filename is empty.");
        return Result::ERROR;
    }

    std::regex filenamePattern("^.+\\.ya?ml$");
    if (!std::regex_match(filename, filenamePattern)) {
        JST_ERROR("[FLOWGRAPH] Invalid filename '{}'.", filename);
        return Result::ERROR;
    }

    _filename = filename;

    return Result::SUCCESS;
}

Result Flowgraph::setTitle(const std::string& title) {
    if (!_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not imported.");
        return Result::ERROR;
    }

    _title = title;

    return Result::SUCCESS;
}

Result Flowgraph::setSummary(const std::string& summary) {
    if (!_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not imported.");
        return Result::ERROR;
    }

    _summary = summary;

    return Result::SUCCESS;
}

Result Flowgraph::setAuthor(const std::string& author) {
    if (!_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not imported.");
        return Result::ERROR;
    }

    _author = author;

    return Result::SUCCESS;
}

Result Flowgraph::setLicense(const std::string& license) {
    if (!_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not imported.");
        return Result::ERROR;
    }

    _license = license;

    return Result::SUCCESS;
}

Result Flowgraph::setDescription(const std::string& description) {
    if (!_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not imported.");
        return Result::ERROR;
    }

    _description = description;

    return Result::SUCCESS;
}

Result Flowgraph::print() const {
    if (!_imported) {
        JST_ERROR("[FLOWGRAPH] Flowgraph is not imported.");
        return Result::ERROR;
    }

    if (_yaml.rootref().empty()) {
        return Result::SUCCESS;
    }

    auto root = _yaml.rootref()[0];
    auto configValues = GatherNodes(root, root, {"protocolVersion",
                                                 "cyberetherVersion"});

    auto optConfigValues = GatherNodes(root, root, {"title",
                                                    "summary",
                                                    "author",
                                                    "license",
                                                    "description"}, true);

    JST_INFO("-----------------------------------------------------------");
    JST_INFO("|                  JETSTREAM CONFIG FILE                  |")
    JST_INFO("-----------------------------------------------------------");
    JST_INFO("Protocol Version:   {}", ResolveReadable(configValues["protocolVersion"]));
    JST_INFO("CyberEther Version: {}", ResolveReadable(configValues["cyberetherVersion"]));
    if (optConfigValues.contains("title")) {
        JST_INFO("Title:              {}", ResolveReadable(optConfigValues["title"]));
    }
    if (optConfigValues.contains("summary")) {
        JST_INFO("Summary:        {}", ResolveReadable(optConfigValues["summary"]));
    }
    if (optConfigValues.contains("author")) {
        JST_INFO("Author:            {}", ResolveReadable(optConfigValues["author"]));
    }
    if (optConfigValues.contains("license")) {
        JST_INFO("License:            {}", ResolveReadable(optConfigValues["license"]));
    }
    if (optConfigValues.contains("description")) {
        JST_INFO("Description:        {}", ResolveReadable(optConfigValues["description"]));
    }

    if (!HasNode(root, root, "graph")) {
        return Result::SUCCESS;
    }

    JST_INFO("------------------------- GRAPH --------------------------");

    for (const auto& node : GetNode(root, root, "graph")) {
        auto values = GatherNodes(root, node, {"module",
                                               "device",
                                               "dataType",
                                               "inputDataType",
                                               "outputDataType"}, true);

        JST_INFO("[{}]:", ResolveReadableKey(node));
        JST_INFO("  Module:            {}", ResolveReadable(values["module"]));
        JST_INFO("  Device:            {}", ResolveReadable(values["device"]));

        if (values.contains("dataType")) {
            JST_INFO("  Data Type:         {}", ResolveReadable(values["dataType"]));
        } else if (values.contains("inputDataType") && values.contains("outputDataType")) {
            JST_INFO("  Data Type:         {} -> {}", ResolveReadable(values["inputDataType"]),
                                                      ResolveReadable(values["outputDataType"]));
        }

        if (HasNode(root, node, "config")) {
            JST_INFO("  Config:");
            for (const auto& element : GetNode(root, node, "config")) {
                JST_INFO("    {} = {}", ResolveReadableKey(element), ResolveReadable(SolvePlaceholder(root, element)));
            }
        }

        if (HasNode(root, node, "input")) {
            JST_INFO("  Input:");
            for (const auto& element : GetNode(root, node, "input")) {
                JST_INFO("    {} = {}", ResolveReadableKey(element), ResolveReadable(SolvePlaceholder(root, element)));
            }
        }

        if (HasNode(root, node, "interface")) {
            JST_INFO("  Interface:");
            for (const auto& element : GetNode(root, node, "interface")) {
                JST_INFO("    {} = {}", ResolveReadableKey(element), ResolveReadable(SolvePlaceholder(root, element)));
            }
        }
    }

    JST_INFO("-----------------------------------------------------------");

    return Result::SUCCESS;
}

}  // namespace Jetstream
