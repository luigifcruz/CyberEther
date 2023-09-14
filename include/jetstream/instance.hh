#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <tuple>
#include <stack>
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "jetstream/state.hh"
#include "jetstream/types.hh"
#include "jetstream/module.hh"
#include "jetstream/bundle.hh"
#include "jetstream/parser.hh"
#include "jetstream/interface.hh"
#include "jetstream/compositor.hh"
#include "jetstream/compute/base.hh"

namespace Jetstream {

class JETSTREAM_API Instance {
 public:
    Instance()
         : _scheduler(),
           _compositor(*this),
           renderState({}),
           viewportState({}), 
           enableFlowgraph(true) {
        JST_DEBUG("[INSTANCE] Started.");
    };

    Result fromFile(const std::string& path);
    Result fromBlob(const char* data);

    Result disableFlowgraph(const bool& disable = true) {
        enableFlowgraph = !disable;
        return Result::SUCCESS;
    }

    template<Device D>
    Result buildBackend(Backend::Config& config) {
        auto& state = backendStates[GetDeviceName(D)];
        state.record.fingerprint.device = GetDeviceName(D);
        JST_CHECK(config>>state.record.configMap);

        return Backend::Initialize<D>(config);
    }

    template<Device D>
    Result buildBackend(Parser::BackendRecord& record) {
        Backend::Config config{};
        JST_CHECK(config<<record.configMap);
        return Backend::Initialize<D>(config);
    }

    template<class Platform, typename... Args>
    Result buildViewport(typename Viewport::Config& config, Args... args) {
        if (_viewport) {
            JST_ERROR("[INSTANCE] A viewport was already created.");
            return Result::ERROR;
        }

        _viewport = std::make_shared<Platform>(config, args...);

        viewportState.record.fingerprint.device = GetDeviceName(_viewport->device());
        viewportState.record.fingerprint.platform = _viewport->name();
        JST_CHECK(config>>viewportState.record.configMap);

        return _viewport->create();
    }

    template<class Platform>
    Result buildViewport(Parser::ViewportRecord& record) {
        typename Viewport::Config config{};
        JST_CHECK(config<<record.configMap);
        return buildViewport<Platform>(config);
    }

    template<Device D>
    Result buildRender(const Render::Window::Config& config) {
        if (!_viewport) {
            JST_ERROR("[INSTANCE] Valid viewport is necessary to create a window.");
            return Result::ERROR;
        }

        if (_window) {
            JST_ERROR("[INSTANCE] A window was already created.");
            return Result::ERROR;
        }

        if (_viewport->device() != D) {
            JST_ERROR("[INSTANCE] Viewport and Window device mismatch.");
            return Result::ERROR;
        }

        auto viewport = std::dynamic_pointer_cast<Viewport::Adapter<D>>(_viewport);
        _window = std::make_shared<Render::WindowImp<D>>(config, viewport);

        JST_CHECK(_window->create());

        return Result::SUCCESS;
    }

    template<Device D>
    Result buildRender(Parser::RenderRecord& record) {
        Render::Window::Config config{};
        JST_CHECK(config<<record.configMap);
        return buildRender<D>(config);
    }

    template<template<Device, typename...> class T, Device D, typename... C>
    Result addModule(std::shared_ptr<T<D, C...>>& module,
                     const std::string& id,
                     const typename T<D, C...>::Config& config,
                     const typename T<D, C...>::Input& input,
                     const std::string& bundleId = "",
                     const Interface::Config& interfaceConfig = {}) {
        using Block = T<D, C...>;

        // Check Block type.

        if constexpr (!std::is_base_of<Interface, Block>::value) {
            JST_ERROR("[INSTANCE] Failed because a module must have an Interface.");
            return Result::ERROR;
        }

        if constexpr (std::is_base_of<Present, Block>::value) {
            if (!_window) {
                JST_ERROR("[INSTANCE] A window is required because "
                          "a graphical module was added.");
                return Result::ERROR;
            }
        }

        // Allocate module.
        module = std::make_shared<Block>();

        // Generate automatic node id if none was provided.
        const auto autoId = [&](){
            if (!id.empty()) {
                return id;
            }

            std::vector<std::string> parts;
            std::string moduleName = module->name();
            std::stringstream ss(moduleName);
            std::string item;

            while (std::getline(ss, item, '-')) {
                parts.push_back(item);
            }

            if (parts.size() < 2) {
                return fmt::format("{}{}", moduleName.substr(0, 3), blockStates.size());
            }

            return fmt::format("{}_{}{}", parts[0].substr(0, 3), parts[1], blockStates.size());
        }();

        // Generate locale and check if it's valid.
        const auto& locale = (!autoId.empty() && !bundleId.empty()) ? Locale{bundleId, autoId} : Locale{autoId};
        JST_DEBUG("[INSTANCE] Adding new module '{}'.", locale);

        if (blockStates.contains(locale)) {
            JST_ERROR("[INSTANCE] Module '{}' already exists.", locale);
            return Result::ERROR;
        }

        // Create new state for module.
        auto& state = blockStates[locale] = std::make_shared<BlockState>();

        // Load generic data.

        module->locale = locale;
        module->config = config;
        module->input = input;

        if constexpr (std::is_base_of<Bundle, Block>::value) {
            module->instance = this;
        }

        // Create module and load state.
        if constexpr (std::is_base_of<Bundle, Block>::value) {
            // Assume bundle is complete. Failed internal modules will mark this bundle incomplete.
            state->complete = true;

            // Create bundle.
            if (module->create() != Result::SUCCESS) {
                // If it fails, mark bundle as incomplete.
                // This is redundant but nice to have I guess.
                state->complete = false;
            }

            // Load state.
            state->bundle = module;
        } else {
            if (!(state->complete = module->create() == Result::SUCCESS)) {
                JST_DEBUG("[INSTANCE] Module '{}' is incomplete.", locale);

                // If module is internal, mark its bundle as incomplete.
                if (locale.internal()) {
                    blockStates.at(locale.idOnly())->complete = false;
                }
            }
            state->module = module;
        }

        // Load state for compute.
        if constexpr (std::is_base_of<Compute, Block>::value) {
            state->compute = module;
        }

        // Load state for present.
        if constexpr (std::is_base_of<Present, Block>::value) {
            if (state->complete) {
                state->present = module;
                state->present->window = _window;
                JST_CHECK(state->present->createPresent());
            }
        }

        // Load state for interface.
        if (!locale.internal()) {
            state->interface = module;
            state->interface->config = interfaceConfig;
        }

        // Populate state record fingerprint.

        state->record.fingerprint.module = module->name();
        state->record.fingerprint.device = GetDeviceName(module->device());

        if constexpr (CountArgs<C...>::value == 1) {
            using Type = typename std::tuple_element<0, std::tuple<C...> >::type;
            state->record.fingerprint.dataType = NumericTypeInfo<Type>::name;
        }
        if constexpr (CountArgs<C...>::value == 2) {
            using InputType = typename std::tuple_element<0, std::tuple<C...> >::type;
            state->record.fingerprint.inputDataType = NumericTypeInfo<InputType>::name;
            using OutputType = typename std::tuple_element<1, std::tuple<C...> >::type;
            state->record.fingerprint.outputDataType = NumericTypeInfo<OutputType>::name;
        }

        // Populate block state record data.

        state->record.locale = locale;

        state->record.setConfigEndpoint(module->config);
        state->record.setInputEndpoint(module->input);
        state->record.setOutputEndpoint(module->output);
        if (!locale.internal()) {
            state->record.setInterfaceEndpoint(state->interface->config);
        }

        JST_CHECK(state->record.updateMaps());

        // Add block to the schedule.
        if (state->complete) {
            JST_CHECK(_scheduler.addModule(locale, state));
        }

        // Add block to the compositor.
        if (!locale.internal()) {
            JST_CHECK(_compositor.addModule(locale, state));
        }

        return Result::SUCCESS;
    }

    template<template<Device, typename...> class T, Device D, typename... C>
    Result addModule(Parser::ModuleRecord& record) {
        using Module = T<D, C...>;

        typename Module::Config config{};
        typename Module::Input input{};
        Interface::Config interfaceConfig{};

        JST_CHECK(config<<record.configMap);
        JST_CHECK(input<<record.inputMap);
        JST_CHECK(interfaceConfig<<record.interfaceMap);

        std::shared_ptr<Module> module;
        const auto& [id, bundleId, _] = record.locale;
        return addModule<T, D, C...>(module, id, config, input, bundleId, interfaceConfig);
    }

    Result eraseModule(const Locale locale);
    Result linkModules(const Locale inputLocale, const Locale outputLocale);
    Result unlinkModules(const Locale inputLocale, const Locale outputLocale);
    Result removeModule(const std::string id, const std::string bundleId = "");
    Result changeModuleBackend(const Locale input, const Device device);
    Result changeModuleDataType(const Locale input, const std::tuple<std::string, std::string, std::string> type);

    Result destroy();
    Result compute();
    Result begin();
    Result present();
    Result end();

    Compositor& compositor() {
        return _compositor;
    }

    Scheduler& scheduler() {
        return _scheduler;
    }

    Render::Window& window() {
        return *_window;
    }

    Viewport::Generic& viewport() {
        return *_viewport;
    }

    Parser& parser() {
        return *_parser;
    }

 protected:
    constexpr RenderState& getRenderState() {
        return renderState;
    }

    bool haveRenderState() {
        return _window != nullptr;
    }

    constexpr ViewportState& getViewportState() {
        return viewportState;
    }

    bool haveViewportState() {
        return _viewport != nullptr;
    }

    BlockState& getBlockState(const std::string& id) {
        return *blockStates[{id}];
    }

    U64 countBlockState(const std::string& id) {
        return blockStates.count({id});
    }

    BackendState& getBackendState(const std::string& key) {
        return backendStates[key];
    }

    U64 countBackendState(const std::string& key) {
        return backendStates.contains(key);
    }

    friend class Parser;

 private:
    Scheduler _scheduler;
    Compositor _compositor;

    RenderState renderState;
    ViewportState viewportState;
    std::unordered_map<std::string, BackendState> backendStates;
    std::unordered_map<Locale, std::shared_ptr<BlockState>, Locale::Hasher> blockStates;

    std::shared_ptr<Render::Window> _window;
    std::shared_ptr<Viewport::Generic> _viewport;
    std::shared_ptr<Parser> _parser;

    bool enableFlowgraph;

    Result fetchDependencyTree(const Locale locale, std::vector<Locale>& storage);

    Result moduleUpdater(const Locale locale, const std::function<Result(const Locale&, Parser::ModuleRecord&)>& updater);
};

}  // namespace Jetstream

#endif
