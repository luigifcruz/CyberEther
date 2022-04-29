#ifndef JETSTREAM_INSTANCE_HH
#define JETSTREAM_INSTANCE_HH

#include <memory>
#include <vector>

#include "jetstream/module.hh"

namespace Jetstream {

class JETSTREAM_API Instance {
 public:
    const Result conduit(const std::vector<std::shared_ptr<Module>>& modules);
    const Result present();
    const Result compute();
      
 private:
    std::atomic_flag computeSync{false};
    std::atomic_flag presentSync{false};

    std::vector<std::shared_ptr<Module>> modules;
};

template<template<Device> class T, Device D>
std::shared_ptr<T<D>> JETSTREAM_API Block(const typename T<D>::Config& config, const typename T<D>::Input& input) {
    return std::make_shared<T<D>>(config, input);
}

Result JETSTREAM_API Conduit(const std::vector<std::shared_ptr<Module>>&);
Result JETSTREAM_API Compute();
Result JETSTREAM_API Present();

}  // namespace Jetstream

#endif
