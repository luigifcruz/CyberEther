#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <memory>
#include <vector>
#include <algorithm>

#include "jetstream/module.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/graph/base.hh"

namespace Jetstream {

class JETSTREAM_API Instance {
 public:
    Instance() : commited(false) {};

    template<class Viewport, typename... Args>
    const Result buildViewport(const typename Viewport::Config& config,
                               Args... args) {
        if (commited) {
            JST_FATAL("The instance was already commited.");
            return Result::ERROR;
        }

        if (_viewport) {
            JST_FATAL("A viewport was already created.");
            return Result::ERROR;
        }

        _viewport = std::make_shared<Viewport>(config, args...);

        return Result::SUCCESS;
    }

    constexpr Viewport::Generic& viewport() {
        return *_viewport;
    }

    template<Device D>
    const Result buildWindow(const Render::Window::Config& config) {
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

        _window = std::make_shared<Render::WindowImp<D>>(config, _viewport);

        return Result::SUCCESS;
    }

    constexpr Render::Window& window() {
        return *_window;
    }

    template<template<Device, typename...> class T, Device D, typename... C>
    std::shared_ptr<T<D, C...>> addBlock(
            const typename T<D, C...>::Config& config,
            const typename T<D, C...>::Input& input) {
        if (commited) {
            JST_FATAL("The instance was already commited.");
            JST_CHECK_THROW(Result::ERROR);
        }

        // Initialize block module.
        auto block = std::make_shared<T<D, C...>>(config, input);

        // Add metadata.
        block->setId(this->blocks.size());

        // Register block module to scheduler.
        this->blocks.push_back(block);

        return block;
    }    

    const Result create();
    const Result destroy();

    const Result compute();

    const Result begin();
    const Result present();
    const Result end();

    constexpr const Viewport::Generic& getViewport() const {
        return *_viewport;
    }

    constexpr const Render::Window& getRender() const {
        return *_window;       
    }

    constexpr const bool isCommited() const {
        return commited;       
    }

 private:
    std::atomic_flag computeSync{false};
    std::atomic_flag presentSync{false};

    std::vector<std::shared_ptr<Module>> blocks;
    std::unordered_map<U64, std::shared_ptr<Present>> presentBlocks;
    std::unordered_map<U64, std::shared_ptr<Compute>> computeBlocks;
        
    std::vector<std::unique_ptr<Graph>> graphs;

    std::shared_ptr<Render::Window> _window;
    std::shared_ptr<Viewport::Generic> _viewport;

    bool commited;
};

}  // namespace Jetstream

#endif
