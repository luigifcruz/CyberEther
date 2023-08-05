#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <tuple>
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "jetstream/state.hh"
#include "jetstream/types.hh"
#include "jetstream/module.hh"
#include "jetstream/bundle.hh"
#include "jetstream/parser.hh"
#include "jetstream/interface.hh"
#include "jetstream/compute/base.hh"

namespace Jetstream {

class JETSTREAM_API Instance {
 public:
    Instance() : renderState({}), viewportState({}), commited(false) {};

    template<Device D>
    Result buildBackend(const Backend::Config& config) {
        auto& state = backendStates[GetDeviceName(D)];
        state.record.id.device = GetDeviceName(D);
        JST_CHECK(config>>state.record.data.configMap);

        return Backend::Initialize<D>(config);
    }

    template<Device D>
    Result buildBackend(Parser::BackendRecord& record) {
        Backend::Config config{};
        JST_CHECK(config<<record.data.configMap);
        return Backend::Initialize<D>(config);
    }

    template<class Platform, typename... Args>
    Result buildViewport(const typename Viewport::Config& config, Args... args) {
        if (commited) {
            JST_FATAL("The instance was already commited.");
            return Result::ERROR;
        }

        if (_viewport) {
            JST_FATAL("A viewport was already created.");
            return Result::ERROR;
        }

        _viewport = std::make_shared<Platform>(config, args...);

        viewportState.record.id.device = GetDeviceName(_viewport->device());
        viewportState.record.id.platform = _viewport->name();
        JST_CHECK(config>>viewportState.record.data.configMap);

        return Result::SUCCESS;
    }

    template<class Platform>
    Result buildViewport(Parser::ViewportRecord& record) {
        typename Viewport::Config config{};
        JST_CHECK(config<<record.data.configMap);
        return buildViewport<Platform>(config);
    }

    template<Device D>
    Result buildRender(const Render::Window::Config& config) {
        if (commited) {
            JST_FATAL("The instance was already commited.");
            return Result::ERROR;
        }

        if (!_viewport) {
            JST_FATAL("Valid viewport is necessary to create a window.");
            return Result::ERROR;
        }

        if (_window) {
            JST_FATAL("A window was already created.");
            return Result::ERROR;
        }

        if (_viewport->device() != D) {
            JST_FATAL("Viewport and Window device mismatch.");
            return Result::ERROR;
        }

        auto viewport = std::dynamic_pointer_cast<Viewport::Adapter<D>>(_viewport);
        _window = std::make_shared<Render::WindowImp<D>>(config, viewport);

        return Result::SUCCESS;
    }

    template<Device D>
    Result buildRender(Parser::RenderRecord& record) {
        Render::Window::Config config{};
        JST_CHECK(config<<record.data.configMap);
        return buildRender<D>(config);
    }

    // TODO: Maybe change this to return a Result instead of throwing.
    template<template<Device, typename...> class T, Device D, typename... C>
    std::shared_ptr<T<D, C...>> addModule(const std::string& name,
                                          const typename T<D, C...>::Config& config,
                                          const typename T<D, C...>::Input& input,
                                          const bool& internalModule = false,
                                          const Interface::Config& interfaceConfig = {}) {
        using Module = T<D, C...>;

        if (commited) {
            JST_FATAL("[INSTANCE] The instance was already commited.");
            JST_CHECK_THROW(Result::ERROR);
        }

        if constexpr (!std::is_base_of<Interface, Module>::value) {
            JST_FATAL("[INSTANCE] Module must have an Interface.");
            JST_CHECK_THROW(Result::ERROR);                
        }

        if (blockStates.contains(name) > 0) {
            JST_FATAL("[INSTANCE] Module name ({}) already exists.", name);
            JST_CHECK_THROW(Result::ERROR);
        }
        auto& blockState = blockStates[name];

        std::shared_ptr<Module> rawBlock;

        if constexpr (std::is_base_of<Bundle, Module>::value) {
            rawBlock = std::make_shared<Module>(*this, name, config, input);
        } else {
            rawBlock = std::make_shared<Module>(config, input);

            if constexpr (std::is_base_of<Compute, Module>::value) {
                blockState.compute = rawBlock;
            }
            if constexpr (std::is_base_of<Present, Module>::value) {
                if (!_window) {
                    JST_ERROR("[INSTANCE] A window is required because "
                              "a graphical block was added.");
                    JST_CHECK_THROW(Result::ERROR);
                }
                blockState.present = rawBlock;
            }

            blockState.module = rawBlock;
        }

        if (!internalModule) {
            blockState.interface = rawBlock;
            blockState.interface->config = interfaceConfig;
        }

        blockState.getConfigFunc = [&](){
            Parser::RecordMap config{};
            JST_CHECK_THROW(rawBlock->getConfig()>>config);
            return config;
        };

        blockState.getInputFunc = [&](){
            Parser::RecordMap input{};
            JST_CHECK_THROW(rawBlock->getInput()>>input);
            return input;
        };

        blockState.getOutputFunc = [&](){
            Parser::RecordMap output{};
            JST_CHECK_THROW(rawBlock->getOutput()>>output);
            return output;
        };

        auto& id = blockState.record.id;
        id.module = rawBlock->name();
        id.device = GetDeviceName(rawBlock->device());

        if constexpr (CountArgs<C...>::value == 1) {
            using Type = typename std::tuple_element<0, std::tuple<C...> >::type;
            id.dataType = NumericTypeInfo<Type>::name;
        }
        if constexpr (CountArgs<C...>::value == 2) {
            using InputType = typename std::tuple_element<0, std::tuple<C...> >::type;
            id.inputDataType = NumericTypeInfo<InputType>::name;
            using OutputType = typename std::tuple_element<1, std::tuple<C...> >::type;
            id.outputDataType = NumericTypeInfo<OutputType>::name;
        }

        blockState.record.data = {
            .configMap = blockState.getConfigFunc(),
            .inputMap = blockState.getInputFunc(),
            .outputMap = blockState.getOutputFunc(),
        };
        blockState.record.name = name;

        blockStateMap[blockStateMap.size()] = name;

        // TODO: Reload engine.

        return rawBlock;
    }
    
    template<template<Device, typename...> class T, Device D, typename... C>
    std::shared_ptr<T<D, C...>> addModule(Parser::ModuleRecord& record,
                                          const bool& internalModule = false) {
        using Module = T<D, C...>;

        typename Module::Config config{};
        typename Module::Input input{};
        Interface::Config interfaceConfig{};

        JST_CHECK_THROW(config<<record.data.configMap);
        JST_CHECK_THROW(input<<record.data.inputMap);
        JST_CHECK_THROW(interfaceConfig<<record.data.interfaceMap);

        return addModule<T, D, C...>(record.name, config, input, internalModule, interfaceConfig);
    }

    // TODO: Add removeModule.
    // TODO: Add rewireModule.

    Result create();
    Result destroy();

    Result compute();

    Result begin();
    Result present();
    Result end();

    Render::Window& window() {
        return *_window;
    }

    Viewport::Generic& viewport() {
        return *_viewport;
    }

    Scheduler& scheduler() {
        return *_scheduler;       
    }

    constexpr bool isCommited() const {
        return commited;       
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
    
    BlockState& getBlockState(const std::string& key) {
        return blockStates[key];
    }

    U64 countBlockState(const std::string& key) {
        return blockStates.contains(key);
    }

    BackendState& getBackendState(const std::string& key) {
        return backendStates[key];
    }

    U64 countBackendState(const std::string& key) {
        return backendStates.contains(key);
    }

    friend class Parser;

 private:
    RenderState renderState;
    ViewportState viewportState;
    std::unordered_map<std::string, BackendState> backendStates;
    std::unordered_map<std::string, BlockState> blockStates;
    std::unordered_map<U64, std::string> blockStateMap;

    std::shared_ptr<Scheduler> _scheduler;
    std::shared_ptr<Render::Window> _window;
    std::shared_ptr<Viewport::Generic> _viewport;

    struct NodeState {
        std::string name;
        std::string title;
        std::unordered_map<U64, std::string> inputs;
        std::unordered_map<U64, std::string> outputs;
    };

    I32 nodeDragId;
    std::unordered_map<U64, NodeState> nodeStates;
    std::vector<std::tuple<U64, U64, U64>> nodeConnections;

    bool commited;
};

}  // namespace Jetstream

#endif
