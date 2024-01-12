#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <tuple>
#include <stack>
#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "jetstream/types.hh"
#include "jetstream/module.hh"
#include "jetstream/block.hh"
#include "jetstream/parser.hh"
#include "jetstream/flowgraph.hh"
#include "jetstream/compositor.hh"
#include "jetstream/compute/base.hh"

// TODO: Add way to disable compositor completely.

namespace Jetstream {

class JETSTREAM_API Instance {
 public:
    Instance()
         : _scheduler(),
           _compositor(*this),
           _flowgraph(*this) {
        JST_DEBUG("[INSTANCE] Started.");
    };

    Result buildInterface(const Device& preferredDevice = Device::None,
                          const Backend::Config& backendConfig = {},
                          const Viewport::Config& viewportConfig = {},
                          const Render::Window::Config& renderConfig = {});

    template<Device D>
    Result buildBackend(const Backend::Config& config) {
        return Backend::Initialize<D>(config);
    }

    template<class Platform, typename... Args>
    Result buildViewport(const typename Viewport::Config& config, Args... args) {
        if (_viewport) {
            JST_ERROR("[INSTANCE] A viewport was already created.");
            return Result::ERROR;
        }

        _viewport = std::make_shared<Platform>(config, args...);

        JST_CHECK(_viewport->create());

        return Result::SUCCESS;
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

    template<template<Device, typename...> class T, Device D, typename... C>
    Result addInternalBlock(std::shared_ptr<T<D, C...>>& block,
                            const std::string& id,
                            const typename T<D, C...>::Config& config,
                            const typename T<D, C...>::Input& input,
                            const Locale& blockLocale = {"main"}) {
        using B = T<D, C...>;

        // Allocate module.

        block = std::make_shared<B>();

        // Build locale.

        Locale locale = blockLocale;

        if (locale.moduleId.empty()) {
            locale.moduleId = id;
        } else {
            locale.moduleId += fmt::format("_{}", id);
        }

        JST_DEBUG("[INSTANCE] Adding internal block '{}'.", locale);

        // Load generic data.

        block->setLocale(locale);
        block->setInstance(this);
        block->setComplete(true);
        block->config = config;
        block->input = input;

        // Create block and load state.

        if (block->create() != Result::SUCCESS) {
            JST_DEBUG("[INSTANCE] Block '{}' is incomplete.", locale);
            block->setComplete(false);
        }

        return Result::SUCCESS;
    }

    template<template<Device, typename...> class T, Device D, typename... C>
    Result eraseInternalBlock(std::shared_ptr<T<D, C...>>& block) {
        if (block->destroy() != Result::SUCCESS) {
            JST_ERROR("[INSTANCE] Failed to destroy internal block '{}'.", block->locale());
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    template<template<Device, typename...> class T, Device D, typename... C>
    Result addModule(std::shared_ptr<T<D, C...>>& module,
                     const std::string& id,
                     const typename T<D, C...>::Config& config,
                     const typename T<D, C...>::Input& input,
                     const Locale& blockLocale = {"main"}) {
        using B = T<D, C...>;

        // Validate module type.

        if constexpr (std::is_base_of<Present, B>::value) {
            if (!_window) {
                JST_FATAL("[INSTANCE] A window is required because "
                          "a graphical module was added.");
                return Result::FATAL;
            }
        }

        if constexpr (!std::is_base_of<Compute, B>::value &&
                      !std::is_base_of<Present, B>::value) {
            JST_FATAL("[INSTANCE] Invalid module invalid because it has "
                      "no compute or present taints.");
            return Result::FATAL;
        }

        // Allocate module.
        module = std::make_shared<B>();

        // Build locale.

        std::string autoId = id;

        if (!blockLocale.moduleId.empty()) {
            autoId = fmt::format("{}_{}", blockLocale.moduleId, id);
        }

        const Locale& locale = {blockLocale.blockId, autoId};
        JST_DEBUG("[INSTANCE] Adding new module '{}'.", locale);
        
        if (_flowgraph.nodes().contains(locale)) {
            JST_ERROR("[INSTANCE] Module '{}' already exists.", locale);
            return Result::ERROR;
        }

        // Create new state for module.

        auto node = std::make_shared<Flowgraph::Node>();
        node->module = module;
        node->id = id;

        // Load generic data.

        module->setLocale(locale);
        module->config = config;
        module->input = input;

        // Create module and load state.

        if (module->create() != Result::SUCCESS) {
            JST_DEBUG("[INSTANCE] Module '{}' is incomplete.", locale);

            auto& block = _flowgraph.nodes().at(locale.block())->block;
            block->setComplete(false);
            block->pushError(fmt::format("[{}] {}", locale, JST_LOG_LAST_ERROR()));

            return Result::SUCCESS;
        }

        // Populate block state record data.

        JST_CHECK(module->input >> node->inputMap);
        JST_CHECK(module->output >> node->outputMap);

        // Check if output locale is owned by module.

        for (const auto& [name, meta] : node->outputMap) {
            if (!meta.locale.empty() && meta.locale.block() != locale.block()) {
                JST_FATAL("[INSTANCE] Module '{}' output '{}' is not owned by the module. "
                          "The output locale is set to '{}'. "
                          "This is likely an internal module error.", locale, name, meta.locale);
                return Result::FATAL;
            }
        }

        // Load state for present.

        if constexpr (std::is_base_of<Present, B>::value) {
            module->window = _window;
            JST_CHECK(module->createPresent());
        }

        // Add module sub-types to nodes.

        if constexpr (std::is_base_of<Compute, B>::value) {
            node->compute = module;
        }

        if constexpr (std::is_base_of<Present, B>::value) {
            node->present = module;
        }

        // Add block to the scheduler.

        JST_CHECK(_scheduler.addModule(locale, 
                                       module, 
                                       node->inputMap, 
                                       node->outputMap, 
                                       node->compute, 
                                       node->present));

        // Add module to the instance.

        _flowgraph.nodes()[locale] = node;
        _flowgraph.nodesOrder().push_back(locale);

        return Result::SUCCESS;
    }

    template<template<Device, typename...> class T, Device D, typename... C>
    Result addBlock(std::shared_ptr<T<D, C...>>& block,
                    const std::string& id,
                    const typename T<D, C...>::Config& config,
                    const typename T<D, C...>::Input& input,
                    const typename T<D, C...>::State& state) {
        using B = T<D, C...>;

        // Allocate module.
        block = std::make_shared<B>();

        // Generate automatic node id if none was provided.

        const auto autoId = [&](){
            if (!id.empty()) {
                return id;
            }

            std::vector<std::string> parts;
            const auto name = std::string(block->id());
            std::stringstream ss(name);
            std::string item;

            while (std::getline(ss, item, '-')) {
                parts.push_back(item);
            }

            if (parts.size() < 2) {
                return fmt::format("{}{}", name.substr(0, 3), _flowgraph.nodes().size());
            }

            return fmt::format("{}_{}{}", parts[0].substr(0, 3), parts[1], _flowgraph.nodes().size());
        }();

        // Build locale.

        const Locale& locale = {autoId};
        JST_DEBUG("[INSTANCE] Adding new block '{}'.", locale);

        if (_flowgraph.nodes().contains(locale)) {
            JST_ERROR("[INSTANCE] Block '{}' already exists.", locale);
            return Result::ERROR;
        }

        // Create new state for block.

        auto node = std::make_shared<Flowgraph::Node>();

        _flowgraph.nodes()[locale] = node;
        _flowgraph.nodesOrder().push_back(locale);

        node->block = block;
        node->id = autoId;

        // Load fingerprint data.

        node->fingerprint.id = block->id();
        node->fingerprint.device = GetDeviceName(block->device());

        if constexpr (CountArgs<C...>::value == 1) {
            using Type = typename std::tuple_element<0, std::tuple<C...> >::type;
            node->fingerprint.inputDataType = NumericTypeInfo<Type>::name;
        }
        if constexpr (CountArgs<C...>::value == 2) {
            using InputType = typename std::tuple_element<0, std::tuple<C...> >::type;
            node->fingerprint.inputDataType = NumericTypeInfo<InputType>::name;
            using OutputType = typename std::tuple_element<1, std::tuple<C...> >::type;
            node->fingerprint.outputDataType = NumericTypeInfo<OutputType>::name;
        }

        // Load generic data.

        block->setLocale(locale);
        block->setInstance(this);
        block->setComplete(true);
        block->state = state;
        block->config = config;
        block->input = input;

        // Create block and load state.

        if (block->create() != Result::SUCCESS) {
            JST_DEBUG("[INSTANCE] Block '{}' is incomplete.", locale);
            block->setComplete(false);
        }

        // Populate block state record data.

        JST_CHECK(block->input >> node->inputMap);
        JST_CHECK(block->output >> node->outputMap);
        JST_CHECK(block->state >> node->stateMap);

        node->setConfigEndpoint(block->config);
        node->setStateEndpoint(block->state);

        // Check if output locale is owned by block.

        for (const auto& [name, meta] : node->outputMap) {
            if (!meta.locale.empty() && meta.locale.block() != locale.block()) {
                JST_FATAL("[INSTANCE] Block '{}' output '{}' is not owned by the block. "
                          "The output locale is set to '{}'. "
                          "This is likely an internal block error.", locale, name, meta.locale);
                return Result::FATAL;
            }
        }
        
        // Add block to the compositor.

        JST_CHECK(_compositor.addBlock(locale, 
                                       block, 
                                       node->inputMap, 
                                       node->outputMap, 
                                       node->stateMap, 
                                       node->fingerprint));

        return Result::SUCCESS;
    }

    template<template<Device, typename...> class T, Device D, typename... C>
    Result addBlock(const std::string& id, 
                    Parser::RecordMap& configMap,
                    Parser::RecordMap& inputMap, 
                    Parser::RecordMap& stateMap) {
        using B = T<D, C...>;

        typename B::Config config{};
        typename B::Input input{};
        typename B::State state{};

        JST_CHECK(config << configMap);
        JST_CHECK(input << inputMap);
        JST_CHECK(state << stateMap);

        std::shared_ptr<B> block;
        return addBlock<T, D, C...>(block, id, config, input, state);
    }

    Result renameBlock(Locale locale, const std::string& id);
    Result removeBlock(Locale locale);
    Result reloadBlock(Locale locale);
    Result linkBlocks(Locale inputLocale, Locale outputLocale);
    Result unlinkBlocks(Locale inputLocale, Locale outputLocale);
    Result changeBlockBackend(Locale input, Device device);
    Result changeBlockDataType(Locale input, std::tuple<std::string, std::string> type);

    Result reset();
    Result eraseModule(Locale locale);
    Result eraseBlock(Locale locale);

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

    Flowgraph& flowgraph() {
        return _flowgraph;
    }

 private:
    Scheduler _scheduler;
    Compositor _compositor;
    Flowgraph _flowgraph;

    std::shared_ptr<Render::Window> _window;
    std::shared_ptr<Viewport::Generic> _viewport;

    Result fetchDependencyTree(Locale locale, std::vector<Locale>& storage);

    Result blockUpdater(Locale locale, const std::function<Result(std::shared_ptr<Flowgraph::Node>&)>& updater);
};

}  // namespace Jetstream

#endif
