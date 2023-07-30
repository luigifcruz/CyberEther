#include "jetstream/parser.hh"

#include "jetstream/store.hh"

namespace Jetstream {

Parser::Parser() {}

Parser::Parser(const std::string& path) {
    _fileData = LoadFile(path);
    _fileTree = ryml::parse_in_place({_fileData.data(), _fileData.size()});
}

Result Parser::importFromFile(Instance& instance) {
    JST_CHECK(createBackends(instance));
    JST_CHECK(createViewport(instance));
    JST_CHECK(createRender(instance));
    JST_CHECK(createModules(instance));

    return Result::SUCCESS;
}

Result Parser::exportToFile(Instance&) {
    // TODO: Implement export.
        
    return Result::SUCCESS;
}

Result Parser::printAll() {
    if (_fileData.empty()) {
        JST_FATAL("[PARSER] Configuration file not loaded.");
        return Result::ERROR;
    }

    auto root = _fileTree.rootref()[0];
    auto configValues = GatherNodes(root, root, {"protocolVersion",
                                                 "cyberetherVersion",
                                                 "name",
                                                 "creator",
                                                 "license"});

    JST_INFO("———————————————————————————————————————————————————————————");
    JST_INFO("|                  JETSTREAM CONFIG FILE                  |")
    JST_INFO("———————————————————————————————————————————————————————————");
    JST_INFO("Protocol Version:   {}", ResolveReadable(configValues["protocolVersion"]));
    JST_INFO("CyberEther Version: {}", ResolveReadable(configValues["cyberetherVersion"]));
    JST_INFO("Name:               {}", ResolveReadable(configValues["name"]));
    JST_INFO("Creator:            {}", ResolveReadable(configValues["creator"]));
    JST_INFO("License:            {}", ResolveReadable(configValues["license"]));

    if (HasNode(root, root, "engine")) {
        auto engine = GetNode(root, root, "engine"); 

        JST_INFO("————————————————————————— ENGINE —————————————————————————");

        if (HasNode(root, engine, "backends")) {
            JST_INFO("Backends:");
            for (const auto& backend : GetNode(root, engine, "backends")) {
                JST_INFO("  Device:           {}", ResolveReadableKey(backend));
                for (const auto& element : backend.children()) {
                    JST_INFO("    {} = {}", ResolveReadableKey(element), ResolveReadable(SolvePlaceholder(root, element)));
                }
            }
        }

        if (HasNode(root, engine, "render")) {
            auto render = GetNode(root, engine, "render");
            JST_INFO("Render:");
            JST_INFO("  Device:           {}", ResolveReadable(render["device"]))
            if (HasNode(root, render, "config")) {
                JST_INFO("  Config:");
                for (const auto& element : GetNode(root, render, "config")) {
                    JST_INFO("    {} = {}", ResolveReadableKey(element), ResolveReadable(SolvePlaceholder(root, element)));
                }
            }
        }

        if (HasNode(root, engine, "viewport")) {
            auto viewport = GetNode(root, engine, "viewport");
            JST_INFO("Viewport:");
            JST_INFO("  Platform:         {}", ResolveReadable(viewport["platform"]))
            if (HasNode(root, viewport, "config")) {
                JST_INFO("  Config:");
                for (const auto& element : GetNode(root, viewport, "config")) {
                    JST_INFO("    {} = {}", ResolveReadableKey(element), ResolveReadable(SolvePlaceholder(root, element)));
                }
            }
        }
    }

    JST_INFO("————————————————————————— GRAPH ——————————————————————————");

    for (const auto& node : GetNode(root, root, "graph")) {
        auto mainValues = GatherNodes(root, node, {"module", "device"});

        JST_INFO("[{}]:", ResolveReadableKey(node));
        JST_INFO("  Module:            {}", ResolveReadable(mainValues["module"]));
        JST_INFO("  Device:            {}", ResolveReadable(mainValues["device"]));

        if (HasNode(root, node, "dataType")) {
            auto typeValues = GatherNodes(root, node, {"dataType"});
            JST_INFO("  Data Type:         {}", ResolveReadable(typeValues["dataType"]));
        } else {
            auto typeValues = GatherNodes(root, node, {"inputDataType", "outputDataType"});
            JST_INFO("  Data Type:         {} -> {}", ResolveReadable(typeValues["inputDataType"]),
                                                      ResolveReadable(typeValues["outputDataType"]));
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
    }
    
    JST_INFO("———————————————————————————————————————————————————————————");

    return Result::SUCCESS;
}

Result Parser::createViewport(Instance& instance) {
    if (_fileData.empty()) {
        JST_FATAL("[PARSER] Configuration file not loaded.");
        return Result::ERROR;
    }

    auto root = _fileTree.rootref()[0];

    if (!HasNode(root, root, "engine")) {
        JST_WARN("[PARSER] No viewport found in configuration file.");
        return Result::SUCCESS;
    }
    auto engine = GetNode(root, root, "engine"); 

    if (!HasNode(root, engine, "viewport")) {
        JST_WARN("[PARSER] No viewport found in configuration file.");
        return Result::SUCCESS;
    }
    auto viewport = GetNode(root, engine, "viewport");

    if (!HasNode(root, engine, "render")) {
        JST_FATAL("[PARSER] No render found in configuration file.");
        return Result::ERROR;
    }
    auto render = GetNode(root, engine, "render");

    ViewportRecord record;

    record.id.platform = ResolveReadable(viewport["platform"]);
    record.id.device = ResolveReadable(render["device"]);

    if (HasNode(root, viewport, "config")) {
        for (const auto& element : GetNode(root, viewport, "config")) {
            auto localPlaceholder = SolvePlaceholder(root, element);
            record.data.configMap[ResolveReadableKey(element)] = {SolveLocalPlaceholder(instance, localPlaceholder), {}};
        }
    }

    return Store::Viewports().at(record.id)(instance, record);
}

Result Parser::createRender(Instance& instance) {
    if (_fileData.empty()) {
        JST_FATAL("[PARSER] Configuration file not loaded.");
        return Result::ERROR;
    }

    auto root = _fileTree.rootref()[0];

    if (!HasNode(root, root, "engine")) {
        JST_WARN("[PARSER] No render found in configuration file.");
        return Result::SUCCESS;
    }
    auto engine = GetNode(root, root, "engine"); 

    if (!HasNode(root, engine, "render")) {
        JST_WARN("[PARSER] No render found in configuration file.");
        return Result::SUCCESS;
    }
    auto render = GetNode(root, engine, "render");

    if (!instance.haveViewportState()) {
        JST_FATAL("[PARSER] No initialized viewport found.");
        return Result::ERROR;
    }

    RenderRecord record;

    record.id.device = ResolveReadable(render["device"]);

    if (HasNode(root, render, "config")) {
        for (const auto& element : GetNode(root, render, "config")) {
            auto localPlaceholder = SolvePlaceholder(root, element);
            record.data.configMap[ResolveReadableKey(element)] = {SolveLocalPlaceholder(instance, localPlaceholder), {}};
        }
    }

    return Store::Renders().at(record.id)(instance, record);

    return Result::SUCCESS;
}

Result Parser::createBackends(Instance& instance) {
    if (_fileData.empty()) {
        JST_FATAL("[PARSER] Configuration file not loaded.");
        return Result::ERROR;
    }

    auto root = _fileTree.rootref()[0];

    if (!HasNode(root, root, "engine")) {
        JST_WARN("[PARSER] No backends found in configuration file.");
        return Result::SUCCESS;
    }
    auto engine = GetNode(root, root, "engine"); 

    if (!HasNode(root, engine, "backends")) {
        JST_WARN("[PARSER] No backends found in configuration file.");
        return Result::SUCCESS;
    }
    auto backends = GetNode(root, engine, "backends");

    for (const auto& backend : backends) {
        BackendRecord record;
        record.id.device = ResolveReadableKey(backend);

        for (const auto& element : backend.children()) {
            auto localPlaceholder = SolvePlaceholder(root, element);
            record.data.configMap[ResolveReadableKey(element)] = {SolveLocalPlaceholder(instance, localPlaceholder), {}};
        }

        JST_CHECK(Store::Backends().at(record.id)(instance, record));
    }

    return Result::SUCCESS;
}

Result Parser::createModules(Instance& instance) {
    if (_fileData.empty()) {
        JST_FATAL("[PARSER] Configuration file not loaded.");
        return Result::ERROR;
    }

    auto root = _fileTree.rootref()[0];
    for (const auto& node : GetNode(root, root, "graph")) {
        const auto nodeKey = ResolveReadableKey(node);
        JST_DEBUG("[PARSER] Processing '{}' module.", nodeKey);

        ModuleRecord record;
        record.name = nodeKey;

        auto mainValues = GatherNodes(root, node, {"module", "device"});
        record.id.module = ResolveReadable(mainValues["module"]);
        record.id.device = ResolveReadable(mainValues["device"]);

        if (HasNode(root, node, "dataType")) {
            auto typeValues = GatherNodes(root, node, {"dataType"});
            record.id.dataType = ResolveReadable(typeValues["dataType"]);
        } else {
            auto typeValues = GatherNodes(root, node, {"inputDataType", "outputDataType"});
            record.id.inputDataType = ResolveReadable(typeValues["inputDataType"]);
            record.id.outputDataType = ResolveReadable(typeValues["outputDataType"]);
        }

        if (HasNode(root, node, "config")) {
            for (const auto& element : GetNode(root, node, "config")) {
                auto localPlaceholder = SolvePlaceholder(root, element);
                record.data.configMap[ResolveReadableKey(element)] = {SolveLocalPlaceholder(instance, localPlaceholder), {}};
            }
        }

        if (HasNode(root, node, "input")) {
            for (const auto& element : GetNode(root, node, "input")) {
                auto localPlaceholder = SolvePlaceholder(root, element);
                record.data.inputMap[ResolveReadableKey(element)] = {SolveLocalPlaceholder(instance, localPlaceholder), {}};
            }
        }

        Store::Modules().at(record.id)(instance, record);
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream