#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <memory>
#include <vector>

#include "jetstream/module.hh"

namespace Jetstream {

class JETSTREAM_API Instance {
 public:
    const Result schedule(const std::shared_ptr<Module>& block);
    const Result present();
    const Result compute();
      
 private:
    std::atomic_flag computeSync{false};
    std::atomic_flag presentSync{false};

    std::vector<std::shared_ptr<Module>> blocks;
};

Result JETSTREAM_API Schedule(const std::shared_ptr<Module>& block);
Result JETSTREAM_API Compute();
Result JETSTREAM_API Present();

template<template<Device, typename...> class T, Device D, typename... C>
std::shared_ptr<T<D, C...>> JETSTREAM_API Block(
        const typename T<D, C...>::Config& config,
        const typename T<D, C...>::Input& input) {
    const auto& block = std::make_shared<T<D, C...>>(config, input);

    if (Schedule(block) != Result::SUCCESS) {
        JST_FATAL("Cannot schedule new block.");
        throw Result::ERROR;
    };

    return block;
}

}  // namespace Jetstream

#endif
