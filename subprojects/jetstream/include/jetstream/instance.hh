#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <memory>
#include <vector>

#include "jetstream/module.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/graph/base.hh"

namespace Jetstream {

class JETSTREAM_API Instance {
 public:
    Instance() : commited(false) {};
    ~Instance();

    template<class Viewport>
    const Result buildViewport(const typename Viewport::Config& config) {
        if (commited) {
            JST_FATAL("The instance was already commited.");
            return Result::ERROR;
        }

        if (_viewport) {
            JST_FATAL("A viewport was already created.");
            return Result::ERROR;
        }

        _viewport = std::make_shared<Viewport>(config);

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

        // Initialize backend for the window.
        if (Backend::Initialize<D>({}) != Result::SUCCESS) {
            JST_FATAL("Cannot initialize window backend.");
            return Result::ERROR;
        }

        _window = std::make_shared<Render::WindowImp<D>>(config, _viewport);

        return Result::SUCCESS;
    }

    constexpr Render::Window& window() {
        return *_window;
    }

    // TODO: Log input and output 
    template<template<Device, typename...> class T, Device D, typename... C>
    std::shared_ptr<T<D, C...>> addBlock(
            const typename T<D, C...>::Config& config,
            const typename T<D, C...>::Input& input) {
        if (commited) {
            JST_FATAL("The instance was already commited.");
            JST_CHECK_THROW(Result::ERROR);
        }

        // Initialize backend for the current block.
        if (Backend::Initialize<D>({}) != Result::SUCCESS) {
            JST_FATAL("Cannot initialize block backend.");
            JST_CHECK_THROW(Result::ERROR);
        }

        // Initialize block module.
        auto block = std::make_shared<T<D, C...>>(config, input);

        // Register block module to scheduler.
        this->blocks.push_back(block);

        return block;
    }    

    const Result commit();

    const Result compute();

    const Result begin();
    const Result present();
    const Result end();

 private:
    std::atomic_flag computeSync{false};
    std::atomic_flag presentSync{false};

    std::vector<std::shared_ptr<Module>> blocks;
    std::vector<std::shared_ptr<Present>> presentBlocks;
    std::vector<std::unique_ptr<Graph>> graphs;

    std::shared_ptr<Render::Window> _window;
    std::shared_ptr<Viewport::Generic> _viewport;

    bool commited;
};

}  // namespace Jetstream

#endif
