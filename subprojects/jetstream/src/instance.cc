#include "jetstream/instance.hh"

namespace Jetstream {

Result Instance::create() {
    JST_DEBUG("Creating instance.");

    // Check if instance isn't commited.
    if (commited) {
        JST_FATAL("The instance was already commited.");
        return Result::ERROR;
    }

    std::vector<std::pair<U64, U64>> nodeInputMap;
    std::unordered_map<U64, U64> nodeOutputMap;

    U64 nodeId = 0;
    for (const auto& [name, block] : blockStates) {
        if (!block.interface) {
            continue;
        }
        auto& state = nodeStates[nodeId++];

        state.name = name;
        state.title = fmt::format("{} ({})", block.interface->prettyName(), name);

        for (const auto& [inputName, inputMeta] : block.record.data.inputMap) {
            nodeInputMap.push_back({inputMeta.vector.phash, nodeId});
            state.inputs.push_back({nodeId++, inputName});
        }

        for (const auto& [outputName, outputMeta] : block.record.data.outputMap) {
            nodeOutputMap[outputMeta.vector.phash] = nodeId;
            state.outputs.push_back({nodeId++, outputName});
        }
    }

    for (const auto& [inputHash, inputId] : nodeInputMap) {
        nodeConnections.push_back({inputId, nodeOutputMap[inputHash]});
    }

    // Create scheduler.
    _scheduler = std::make_shared<Scheduler>(_window, blockStates, blockStateMap);

    // Initialize instance window.
    if (_window && _viewport) {
        JST_CHECK(_viewport->create());
        JST_CHECK(_window->create());
    }

    // Lock instance after initialization.
    commited = true;

    return Result::SUCCESS;
}

Result Instance::destroy() {
    JST_DEBUG("Destroying instance.");

    // Check if instance is commited.
    if (!commited) {
        JST_FATAL("Can't create instance that wasn't created.");
        return Result::ERROR;
    }

    // Destroy instance window.
    if (_window && _viewport) {
        JST_CHECK(_window->destroy());
        JST_CHECK(_viewport->destroy());
    }

    // Destroy scheduler.
    _scheduler.reset();

    // Unlock instance.
    commited = false;

    return Result::SUCCESS;
}

Result Instance::compute() {
    return _scheduler->compute();
}

Result Instance::begin() {
    if (_window) {
        JST_CHECK(_window->begin());
    }

    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

    return Result::SUCCESS;
}

Result Instance::present() {
    return _scheduler->present();
}

Result Instance::end() {
    for (auto& [_, bundle] : blockStates) {
        if (!bundle.interface) {
            continue;
        }
        JST_CHECK(bundle.interface->drawView());
    }

    ImGui::Begin("Control");
    for (auto& [_, bundle] : blockStates) {
        if (!bundle.interface) {
            continue;
        }
        JST_CHECK(bundle.interface->drawControl());
    }
    ImGui::End();

    ImGui::Begin("Info");
    for (auto& [_, bundle] : blockStates) {
        if (!bundle.interface) {
            continue;
        }
        JST_CHECK(bundle.interface->drawInfo());
    }
    ImGui::End();

    ImGui::Begin("Flowgraph");
    ImNodes::BeginNodeEditor();
    for (const auto& [id, state] : nodeStates) {
        ImNodes::BeginNode(id);

        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(state.title.c_str());
        ImNodes::EndNodeTitleBar();

        ImGui::Dummy(ImVec2(80.0f, 45.0f));

        JST_CHECK(blockStates[state.name].interface->drawNodeControl());

        for (const auto& [inputId, inputName] : state.inputs) {
            ImNodes::BeginInputAttribute(inputId);
            ImGui::Text(inputName.c_str());
            ImNodes::EndInputAttribute();
        }

        for (const auto& [outputId, outputName] : state.outputs) {
            ImNodes::BeginOutputAttribute(outputId);
            ImGui::Text(outputName.c_str());
            ImNodes::EndInputAttribute();
        }

        JST_CHECK(blockStates[state.name].interface->drawNode());

        ImNodes::EndNode();
    }

    U64 nodeConnectionId = 0;
    for (const auto& [a, b] : nodeConnections) {
        ImNodes::Link(nodeConnectionId++, a, b);
    }

    ImNodes::EndNodeEditor();
    ImGui::End();

    if (_window) {
        JST_CHECK(_window->end());
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream
