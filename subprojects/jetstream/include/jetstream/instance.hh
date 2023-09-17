#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <memory>
#include <vector>
#include <algorithm>
#include <unordered_map>

#include "jetstream/module.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/render/base.hh"
#include "jetstream/graph/base.hh"

namespace Jetstream {

class JETSTREAM_API Instance {
 public:
    Instance() : commited(false) {};

    template<class Platform, typename... Args>
    Result buildViewport(const typename Viewport::Config& config,
                         Args... args) {
        if (commited) {
            JST_FATAL("The instance was already commited.");
            return Result::ERROR;
        }

        if (_viewport) {
            JST_FATAL("A viewport was already created.");
            return Result::ERROR;
        }

        _viewport = std::make_shared<Platform>(config, args...);

        return Result::SUCCESS;
    }

    Viewport::Generic& viewport() {
        return *_viewport;
    }

    template<Device D>
    Result buildWindow(const Render::Window::Config& config) {
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

    Render::Window& window() {
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

    template<template<Device, typename...> class T, Device D, typename... C>
    Result addBlock(std::shared_ptr<T<D, C...>>& block) {
        if (commited) {
            JST_FATAL("The instance was already commited.");
            JST_CHECK_THROW(Result::ERROR);
        }

        // Add metadata.
        block->setId(this->blocks.size());

        // Register block module to scheduler.
        this->blocks.push_back(block);

        return Result::SUCCESS;
    }

    Result create();
    Result destroy();

    Result compute();

    Result begin();
    Result present();
    Result end();

    const Viewport::Generic& getViewport() const {
        return *_viewport;
    }

    const Render::Window& getRender() const {
        return *_window;       
    }

    bool isCommited() const {
        return commited;       
    }

 private:
    std::atomic_flag computeSync = ATOMIC_FLAG_INIT;
    std::atomic_flag presentSync = ATOMIC_FLAG_INIT;

    std::vector<std::shared_ptr<Module>> blocks;

    std::unordered_map<U64, std::shared_ptr<Present>> presentBlocks;
    std::unordered_map<U64, std::shared_ptr<Compute>> computeBlocks;

    std::unordered_map<U64, std::vector<U64>> blockInputs, blockOutputs;
    std::unordered_map<U64, std::vector<U64>> blockInputsPos, blockOutputsPos;

    std::vector<U64> executionOrder;
    std::vector<std::pair<Device, std::vector<U64>>> deviceExecutionOrder;

    std::vector<std::shared_ptr<Graph>> graphs;

    std::shared_ptr<Render::Window> _window;
    std::shared_ptr<Viewport::Generic> _viewport;

    Result printGraphSummary();
    Result filterStaleIo();
    Result applyTopologicalSort();
    Result createComputeGraphs();
    Result assertInplaceCorrectness();

    bool commited;
};

}  // namespace Jetstream

#endif
