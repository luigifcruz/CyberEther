#ifndef JETSTREAM_BACKEND_BASE_HH
#define JETSTREAM_BACKEND_BASE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/backend/config.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/backend/devices/cpu/base.hh"
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/backend/devices/metal/base.hh"
#endif

namespace Jetstream::Backend {

class JETSTREAM_API Instance {
 public:
    template<Device D>
    const Result initialize(const Config& config);

    template<Device D>
    const Result destroy();

    template<Device D>
    const auto& state();

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    template<>
    const Result initialize<Device::CPU>(const Config& config) {
        if (!cpu) {
            JST_DEBUG("Initializing CPU backend.");
            cpu = std::make_unique<CPU>(config);
        }
        return Result::SUCCESS;
    }

    template<>
    const Result destroy<Device::CPU>() {
        if (cpu) {
            cpu.reset();
        }
        return Result::SUCCESS;
    }

    template<>
    const auto& state<Device::CPU>() {
        return cpu; 
    }
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    template<>
    const Result initialize<Device::Metal>(const Config& config) {
        if (!metal) {
            JST_DEBUG("Initializing Metal backend.");
            metal = std::make_unique<Metal>(config);
        }
        return Result::SUCCESS;
    }

    template<>
    const Result destroy<Device::Metal>() {
        if (metal) {
            metal.reset();
        }
        return Result::SUCCESS;
    }

    template<>
    const auto& state<Device::Metal>() {
        return metal; 
    }
#endif

 private:
#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
    std::unique_ptr<CPU> cpu;
#endif
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
    std::unique_ptr<Metal> metal;
#endif
};

Instance& JETSTREAM_API Get();

template<Device D>
const auto& JETSTREAM_API State() {
    return Get().state<D>();
}

template<Device D>
const Result JETSTREAM_API Initialize(const Config& config) {
    return Get().initialize<D>(config);
}

template<Device D>
const Result JETSTREAM_API Destroy() {
    return Get().destroy<D>();
}

}  // namespace Jetstream::Backend

#endif
