#include "jetstream/instance.hh"
#include "jetstream/store.hh"

namespace Jetstream {

Result Instance::openFlowgraphFile(const std::string& path) {
    if (_parser->haveFlowgraph()) {
        JST_ERROR("[INSTANCE] Flowgraph already loaded.");
        return Result::ERROR;
    }

    if (path.empty()) {
        JST_ERROR("[INSTANCE] Configuration path is empty.");
        return Result::ERROR;
    }
    
    JST_DEBUG("[INSTANCE] Importing flowgraph from file ({}).", path);

    JST_CHECK(_parser->openFlowgraphFile(path));
    JST_CHECK(_parser->printFlowgraph());
    JST_CHECK(_parser->importFlowgraph(*this));

    return Result::SUCCESS;
}

Result Instance::openFlowgraphBlob(const char* blob) {
    if (_parser->haveFlowgraph()) {
        JST_ERROR("[INSTANCE] Flowgraph already loaded.");
        return Result::ERROR;
    }

    JST_DEBUG("[INSTANCE] Importing flowgraph from blob.");

    JST_CHECK(_parser->openFlowgraphBlob(blob));
    JST_CHECK(_parser->printFlowgraph());
    JST_CHECK(_parser->importFlowgraph(*this));

    return Result::SUCCESS;
}

Result Instance::saveFlowgraph(const std::string& path) {
    if (!_parser->haveFlowgraph()) {
        JST_ERROR("[INSTANCE] Flowgraph not loaded.");
        return Result::ERROR;
    }

    JST_CHECK(_parser->exportFlowgraph(*this));
    JST_CHECK(_parser->saveFlowgraph(path));

    return Result::SUCCESS;
}

Result Instance::resetFlowgraph() {
    if (!_parser->haveFlowgraph()) {
        JST_ERROR("[INSTANCE] Flowgraph not loaded.");
        return Result::ERROR;
    }

    JST_CHECK(clearModules());

    JST_CHECK(_parser->exportFlowgraph(*this));

    return Result::SUCCESS;
}

Result Instance::closeFlowgraph() {
    JST_CHECK(clearModules());

    JST_CHECK(_parser->closeFlowgraph());

    return Result::SUCCESS;
}

Result Instance::newFlowgraph() {
    if (_parser->haveFlowgraph()) {
        JST_ERROR("[INSTANCE] Flowgraph already loaded.");
        return Result::ERROR;
    }

    JST_CHECK(closeFlowgraph());
    JST_CHECK(openFlowgraphBlob("\n"));

    return Result::SUCCESS;
}

Result Instance::buildDefaultInterface() {
    JST_DEBUG("[INSTANCE] Building default viewport and render.");

    // Default Viewport configuration.
    Viewport::Config viewportCfg;
    viewportCfg.vsync = true;
    viewportCfg.size = {3000, 2000};
    viewportCfg.title = "CyberEther";

    // Default Window configuration.
    Render::Window::Config renderCfg;
    renderCfg.imgui = true;
    renderCfg.scale = 1.0;

    if (!_viewport) {
#ifdef JETSTREAM_VIEWPORT_GLFW_AVAILABLE
#if   defined(JETSTREAM_BACKEND_METAL_AVAILABLE)
        JST_CHECK(this->buildViewport<Viewport::GLFW<Device::Metal>>(viewportCfg));
#elif defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE)
        JST_CHECK(this->buildViewport<Viewport::GLFW<Device::Vulkan>>(viewportCfg));
#elif defined(JETSTREAM_BACKEND_WEBGPU_AVAILABLE)
        JST_CHECK(this->buildViewport<Viewport::GLFW<Device::WebGPU>>(viewportCfg));
#endif
#else
        JST_ERROR("[INSTANCE] No viewport backend available.");
        return Result::ERROR;
#endif
    }

    if (!_window) {
#if   defined(JETSTREAM_BACKEND_METAL_AVAILABLE)
        JST_CHECK(this->buildRender<Device::Metal>(renderCfg));
#elif defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE)
        JST_CHECK(this->buildRender<Device::Vulkan>(renderCfg));
#elif defined(JETSTREAM_BACKEND_WEBGPU_AVAILABLE)
        JST_CHECK(this->buildRender<Device::WebGPU>(renderCfg));
#else
        JST_ERROR("[INSTANCE] No render backend available.");
        JST_CHECK(Result::ERROR);
#endif
    }

    return Result::SUCCESS;
}

Result Instance::reloadModule(const Locale locale) {
    JST_DEBUG("[INSTANCE] Reloading module '{}'.", locale);

    if (!blockStates.contains(locale)) {
        JST_ERROR("[INSTANCE] Module '{}' doesn't exist.", locale);
        return Result::ERROR;
    }

    JST_CHECK(moduleUpdater(locale, [](const Locale&, Parser::ModuleRecord&){
        return Result::SUCCESS;
    }));

    return Result::SUCCESS;
}

Result Instance::removeModule(const std::string id, const std::string bundleId) {
    const auto& locale = (!id.empty() && !bundleId.empty()) ? Locale{bundleId, id} : Locale{id};

    JST_DEBUG("[INSTANCE] Trying to remove module '{}'.", locale);

    // Verify input parameters.

    if (!blockStates.contains(locale)) {
        JST_ERROR("[INSTANCE] Module '{}' doesn't exist.", locale);
        return Result::ERROR;
    }

    // Unlink all modules linked to the module being removed.

    std::vector<std::pair<Locale, Locale>> unlinkList;
    // TODO: This is exponential garbage. Implement cached system.
    for (const auto& [outputPinId, outputRecord] : blockStates.at(locale)->record.outputMap) {
        for (const auto& [inputLocale, inputState] : blockStates) {
            for (const auto& [inputPinId, inputRecord] : inputState->record.inputMap) {
                if (inputRecord.locale == outputRecord.locale) {
                    unlinkList.push_back({{inputLocale.id, inputLocale.subId, inputPinId}, outputRecord.locale});
                }
            }
        }
    }

    for (const auto& [inputLocale, outputLocale] : unlinkList) {
        JST_CHECK(unlinkModules(inputLocale, outputLocale));
    }

    // Delete module.
    JST_CHECK(eraseModule(locale));

    return Result::SUCCESS;
}

Result Instance::unlinkModules(const Locale inputLocale, const Locale outputLocale) {
    JST_DEBUG("[INSTANCE] Unlinking '{}' -> '{}'.", outputLocale, inputLocale);

    // Verify input parameters.

    if (!blockStates.contains(inputLocale.idOnly())) {
        JST_ERROR("[INSTANCE] Link input block '{}' doesn't exist.", inputLocale);
        return Result::ERROR;
    }

    if (!blockStates.contains(outputLocale.idOnly())) {
        JST_ERROR("[INSTANCE] Link output block '{}' doesn't exist.", outputLocale);
        return Result::ERROR;
    }

    if (inputLocale.pinId.empty() || outputLocale.pinId.empty()) {
        JST_ERROR("[INSTANCE] The pin ID of the input and output locale must not be empty.");
        return Result::ERROR;
    }

    const auto& rawInputLocale = blockStates.at(inputLocale.idOnly())->record.inputMap.at(inputLocale.pinId).locale;
    if (rawInputLocale.idOnly() != outputLocale.idOnly()) {
        JST_ERROR("[INSTANCE] Link '{}' -> '{}' doesn't exist.", outputLocale, inputLocale);
        return Result::ERROR;
    }

    // Update module.

    JST_CHECK(moduleUpdater(inputLocale, [&](const Locale&, Parser::ModuleRecord& record) {
        // Delete link from input block inputs.
        record.inputMap.erase(inputLocale.pinId);

        return Result::SUCCESS;
    }));

    return Result::SUCCESS;
}

Result Instance::linkModules(const Locale inputLocale, const Locale outputLocale) {
    JST_DEBUG("[INSTANCE] Linking '{}' -> '{}'.", outputLocale, inputLocale);

    // Verify input parameters.

    if (!blockStates.contains(inputLocale.idOnly())) {
        JST_ERROR("[INSTANCE] Link input block '{}' doesn't exist.", inputLocale);
        return Result::ERROR;
    }

    if (!blockStates.contains(outputLocale.idOnly())) {
        JST_ERROR("[INSTANCE] Link output block '{}' doesn't exist.", outputLocale);
        return Result::ERROR;
    }

    if (inputLocale.pinId.empty() || outputLocale.pinId.empty()) {
        JST_ERROR("[INSTANCE] The pin ID of the input and output locale must not be empty.");
        return Result::ERROR;
    }

    const auto& rawInputLocale = blockStates.at(inputLocale.idOnly())->record.inputMap.at(inputLocale.pinId).locale;
    if (rawInputLocale.idOnly() == outputLocale.idOnly()) {
        JST_ERROR("[INSTANCE] Link '{}' -> '{}' already exists.", outputLocale, inputLocale);
        return Result::ERROR;
    }

    const auto& inputRecordMap = blockStates.at(inputLocale.idOnly())->record.inputMap;
    if (inputRecordMap.contains(inputLocale.pinId) && inputRecordMap.at(inputLocale.pinId).hash != 0) {
        JST_ERROR("[INSTANCE] Input '{}' is already linked with something else.", inputLocale);
        return Result::ERROR;
    }

    // Update module.

    JST_CHECK(moduleUpdater(inputLocale, [&](const Locale& locale, Parser::ModuleRecord& record) {
        // Delete link from input block inputs.
        auto& outputMap = blockStates.at(outputLocale.idOnly())->record.outputMap;
        record.inputMap[locale.pinId] = outputMap.at(outputLocale.pinId);

        return Result::SUCCESS;
    }));

    return Result::SUCCESS;
}

Result Instance::changeModuleBackend(const Locale input, const Device device) {
    JST_DEBUG("[INSTANCE] Changing module '{}' backend to '{}'.", input, device);

    // Verify input parameters.

    if (!blockStates.contains(input.idOnly())) {
        JST_ERROR("[INSTANCE] Module '{}' doesn't exist.", input);
        return Result::ERROR;
    }

    // Update module.

    JST_CHECK(moduleUpdater(input, [&](const Locale&, Parser::ModuleRecord& record) {
        // Change backend.
        record.fingerprint.device = GetDeviceName(device);

        return Result::SUCCESS;
    }));

    return Result::SUCCESS;
}

Result Instance::changeModuleDataType(const Locale input, const std::tuple<std::string, std::string, std::string> type) {
    JST_DEBUG("[INSTANCE] Changing module '{}' data type to '{}'.", input, type);

    // Verify input parameters.

    if (!blockStates.contains(input.idOnly())) {
        JST_ERROR("[INSTANCE] Module '{}' doesn't exist.", input);
        return Result::ERROR;
    }

    // Update module.

    JST_CHECK(moduleUpdater(input, [&](const Locale&, Parser::ModuleRecord& record) {
        // Change data type.
        const auto& [dataType, inputDataType, outputDataType] = type;
        record.fingerprint.dataType = dataType;
        record.fingerprint.inputDataType = inputDataType;
        record.fingerprint.outputDataType = outputDataType;

        // Clean-up every state because they will most likely be incompatible.
        record.inputMap.clear();
        record.outputMap.clear();
        record.configMap.clear();

        return Result::SUCCESS;
    }));

    return Result::SUCCESS;
}

Result Instance::moduleUpdater(const Locale locale, const std::function<Result(const Locale&, Parser::ModuleRecord&)>& updater) {
    // List all dependencies.

    std::vector<Locale> dependencyTree;
    JST_CHECK(fetchDependencyTree(locale.idOnly(), dependencyTree));

    // Copy and update dependency blocks states and then delete.

    std::vector<Parser::ModuleRecord> dependencyRecords;
    for (const auto& dependencyLocale : dependencyTree | std::ranges::views::reverse) {
        // Skip if dependency is internal.
        if (dependencyLocale.internal()) {
            continue;
        }

        // Fetch dependency record.
        auto& record = blockStates.at(dependencyLocale)->record;

        // Update dependency block records.
        JST_CHECK(record.updateMaps());

        // Copy dependency block records.
        dependencyRecords.push_back(record);

        // Delete dependency block.
        JST_CHECK(eraseModule(dependencyLocale));
    }

    // Copy input block record.
    auto record = blockStates.at(locale.idOnly())->record;

    // Update input block records.
    JST_CHECK(record.updateMaps());

    // Delete input block.
    JST_CHECK(eraseModule(locale.idOnly()));

    // Backup original record.
    auto recordBackup = record;

    // Call user defined record updater.
    JST_CHECK(updater(locale, record));

    auto res = Result::SUCCESS;

    // Check if the module store has such a record fingerprint.
    if (!Store::Modules().contains(record.fingerprint)) {
        JST_ERROR("[INSTANCE] Module fingerprint doesn't exist.");        
        res = Result::ERROR;
    } else {
        res = Store::Modules().at(record.fingerprint)(*this, record);
    }

    // Create new input block using saved records.
    const auto errorCodeBackup = JST_LOG_LAST_ERROR();

    // If recreation fails, rewind the module state.
    if (res != Result::SUCCESS) {
        JST_CHECK(Store::Modules().at(recordBackup.fingerprint)(*this, recordBackup));
    }

    // Create new dependency block using saved records.
    for (auto& record : dependencyRecords | std::ranges::views::reverse) {
        // Update input of dependency block record.
        for (auto& [inputName, inputRecord] : record.inputMap) {
            auto& outputRecord = blockStates.at(inputRecord.locale.idOnly())->record;
            inputRecord = outputRecord.outputMap.at(inputRecord.locale.pinId);
        }

        // Create new dependency block with updated inputs.
        JST_CHECK(Store::Modules().at(record.fingerprint)(*this, record));
    }

    JST_LOG_LAST_ERROR() = errorCodeBackup;

    return res;
}

Result Instance::eraseModule(const Locale locale) {
    // Verify input parameters.
    if (!blockStates.contains(locale)) {
        JST_ERROR("[INSTANCE] Module '{}' doesn't exist.", locale);
        return Result::ERROR;
    }

    // Remove module from state.
    auto state = blockStates.extract(locale).mapped();

    // Remove block from schedule.
    if (state->complete) {
        JST_CHECK(_scheduler.removeModule(locale));
    }

    // Remove block from compositor.
    if (!locale.internal()) {
        JST_CHECK(_compositor.removeModule(locale));
    }

    // Destroy the present logic.
    if (state->present) {
        JST_CHECK(state->present->destroyPresent());
    }

    // Destroy the module or bundle.
    if (state->module) {
        JST_CHECK(state->module->destroy());
    } else if (state->bundle) {
        JST_CHECK(state->bundle->destroy());
    }

    return Result::SUCCESS;
}

Result Instance::fetchDependencyTree(const Locale locale, std::vector<Locale>& storage) {
    std::stack<Locale> stack;
    std::unordered_set<Locale, Locale::Hasher> seenLocales;

    stack.push(locale);
    seenLocales.insert(locale);

    while (!stack.empty()) {
        Locale currentLocale = stack.top();
        stack.pop();

        // TODO: This is an exponential garbage hack. Implement cached system.
        for (const auto& [outputPinId, outputRecord] : blockStates.at(currentLocale)->record.outputMap) {
            for (const auto& [inputLocale, inputState] : blockStates) {
                for (const auto& [inputPinId, inputRecord] : inputState->record.inputMap) {
                    if (inputRecord.locale == outputRecord.locale) {
                        Locale nextLocale = inputLocale.idOnly();
                        if (seenLocales.find(nextLocale) == seenLocales.end()) {
                            storage.push_back(nextLocale);
                            stack.push(nextLocale);
                            seenLocales.insert(nextLocale);
                        }
                    }
                }
            }
        }
    }

    return Result::SUCCESS;
}

Result Instance::clearModules() {
    JST_DEBUG("[INSTANCE] Clearing modules from instance.");

    // Destroying compositor and scheduler.
    JST_CHECK(_compositor.destroy());
    JST_CHECK(_scheduler.destroy());

    // Destroy all present modules.
    for (auto& [_, state] : blockStates) {
        if (!state->present) {
            continue;
        }
        JST_CHECK(state->present->destroyPresent());
    }

    // Destroy all modules.
    for (auto& [_, state] : blockStates) {
        // TODO: This ignores bundles. Is this right?
        if (!state->module) {
            continue;
        }
        JST_CHECK(state->module->destroy());
    }

    // Clear internal state memory.
    blockStates.clear();

    return Result::SUCCESS;
}

Result Instance::destroy() {
    JST_DEBUG("[INSTANCE] Destroying instance.");

    // Clear modules.
    JST_CHECK(clearModules());

    // Destroy window and viewport.

    if (_window) {
        JST_CHECK(_window->destroy());
    }

    if (_viewport) {
        JST_CHECK(_viewport->destroy());
    }

    _parser = std::make_shared<Parser>();

    return Result::SUCCESS;
}

Result Instance::compute() {
    return _scheduler.compute();
}

Result Instance::begin() {
    // Create new render frame.
    JST_CHECK(_window->begin());

    return Result::SUCCESS;
}

Result Instance::present() {
    // Update the modules present logic.
    JST_CHECK(_scheduler.present());

    return Result::SUCCESS;
}

Result Instance::end() {
    // Draw the main interface.
    JST_CHECK(_compositor.draw());

    // Finish the render frame.
    JST_CHECK(_window->end());

    // Process interactions after finishing frame.
    JST_CHECK(_compositor.processInteractions());

    return Result::SUCCESS;
}

}  // namespace Jetstream
