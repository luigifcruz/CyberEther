#include "jetstream/parser.hh"

#include "jetstream/store.hh"

namespace Jetstream {

Parser::Parser() {}

Result Parser::openFlowgraphFile(const std::string& path) {
    _fileData = LoadFile(path);
    _fileTree = ryml::parse_in_place({_fileData.data(), _fileData.size()});

    return Result::SUCCESS;
}

Result Parser::openFlowgraphBlob(const char* blob) {
    _fileData.insert(_fileData.begin(), blob, blob + strlen(blob));
    
    // Add a newline at the end of the file in case of an empty file.
    if (_fileData.empty() || _fileData.back() != '\n') {
        _fileData.push_back('\n');
    }

    _fileTree = ryml::parse_in_place({_fileData.data(), _fileData.size()});

    return Result::SUCCESS;
}

Result Parser::printFlowgraph() const {
    if (_fileData.empty()) {
        JST_ERROR("[PARSER] Configuration file not loaded.");
        return Result::ERROR;
    }

    if (_fileTree.rootref().empty()) {
        return Result::SUCCESS;
    }

    auto root = _fileTree.rootref()[0];
    auto configValues = GatherNodes(root, root, {"protocolVersion",
                                                 "cyberetherVersion"});

    auto optConfigValues = GatherNodes(root, root, {"title",
                                                    "description",
                                                    "creator",
                                                    "license"}, true);

    JST_INFO("———————————————————————————————————————————————————————————");
    JST_INFO("|                  JETSTREAM CONFIG FILE                  |")
    JST_INFO("———————————————————————————————————————————————————————————");
    JST_INFO("Protocol Version:   {}", ResolveReadable(configValues["protocolVersion"]));
    JST_INFO("CyberEther Version: {}", ResolveReadable(configValues["cyberetherVersion"]));
    if (optConfigValues.contains("title")) {
        JST_INFO("Title:              {}", ResolveReadable(optConfigValues["title"]));
    }
    if (optConfigValues.contains("description")) {
        JST_INFO("Description:        {}", ResolveReadable(optConfigValues["description"]));
    }
    if (optConfigValues.contains("creator")) {
        JST_INFO("Creator:            {}", ResolveReadable(optConfigValues["creator"]));
    }
    if (optConfigValues.contains("license")) {
        JST_INFO("License:            {}", ResolveReadable(optConfigValues["license"]));
    }

    JST_INFO("————————————————————————— GRAPH ——————————————————————————");

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

    JST_INFO("———————————————————————————————————————————————————————————");

    return Result::SUCCESS;
}

Result Parser::importFlowgraph(Instance& instance) {
    if (_fileData.empty()) {
        JST_ERROR("[PARSER] Configuration file not loaded.");
        return Result::ERROR;
    }

    if (_fileTree.rootref().empty()) {
        return Result::SUCCESS;
    }

    auto root = _fileTree.rootref()[0];
    for (const auto& node : GetNode(root, root, "graph")) {
        const auto nodeKey = ResolveReadableKey(node);
        JST_DEBUG("[PARSER] Processing '{}' module.", nodeKey);

        ModuleRecord record;

        // Populate fingerprint.

        auto values = GatherNodes(root, node, {"module", "device", "dataType", "inputDataType", "outputDataType"}, true);
        record.fingerprint.module = ResolveReadable(values["module"]);
        record.fingerprint.device = ResolveReadable(values["device"]);
        if (values.contains("dataType")) {
            record.fingerprint.dataType = ResolveReadable(values["dataType"]);
        } else if (values.contains("inputDataType") && values.contains("outputDataType")) {
            record.fingerprint.inputDataType = ResolveReadable(values["inputDataType"]);
            record.fingerprint.outputDataType = ResolveReadable(values["outputDataType"]);
        }

        // Populate data.

        record.locale = {nodeKey};

        if (HasNode(root, node, "config")) {
            for (const auto& element : GetNode(root, node, "config")) {
                auto localPlaceholder = SolvePlaceholder(root, element);
                record.configMap[ResolveReadableKey(element)] = SolveLocalPlaceholder(instance, localPlaceholder);
            }
        }

        if (HasNode(root, node, "input")) {
            for (const auto& element : GetNode(root, node, "input")) {
                auto localPlaceholder = SolvePlaceholder(root, element);
                record.inputMap[ResolveReadableKey(element)] = SolveLocalPlaceholder(instance, localPlaceholder);
            }
        }

        if (HasNode(root, node, "interface")) {
            for (const auto& element : GetNode(root, node, "interface")) {
                auto localPlaceholder = SolvePlaceholder(root, element);
                record.interfaceMap[ResolveReadableKey(element)] = SolveLocalPlaceholder(instance, localPlaceholder);
            }
        }

        if (!Store::Modules().contains(record.fingerprint)) {
            JST_ERROR("[PARSER] Can't find module with such a signature ({}).", record.fingerprint);
            return Result::ERROR;
        }

        JST_CHECK(Store::Modules().at(record.fingerprint)(instance, record));
    }

    return Result::SUCCESS;
}

Result Parser::exportFlowgraph(Instance&) {
    // TODO: Implement.

    return Result::SUCCESS;
}

Result Parser::saveFlowgraph(const std::string&) {
    // TODO: Implement.

    return Result::SUCCESS;
}

Result Parser::closeFlowgraph() {
    _fileData.clear();

    return Result::SUCCESS;
}

}  // namespace Jetstream
