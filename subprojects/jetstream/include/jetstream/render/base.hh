#ifndef JETSTREAM_RENDER_BASE_HH
#define JETSTREAM_RENDER_BASE_HH

#include "jetstream/logger.hh"
#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/render/types.hh"

#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/base/draw.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/render/base/program.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/vertex.hh"

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
#include "jetstream/render/metal/window.hh"
#include "jetstream/render/metal/surface.hh"
#include "jetstream/render/metal/program.hh"
#include "jetstream/render/metal/buffer.hh"
#include "jetstream/render/metal/draw.hh"
#include "jetstream/render/metal/texture.hh"
#include "jetstream/render/metal/vertex.hh"
#endif

namespace Jetstream::Render {

template<class T>
inline Result Create(const Device& device, 
                                   std::shared_ptr<T>& member, 
                                   const auto& config) {
    switch (device) {
        case Device::Metal:
            member = T::template Factory<Device::Metal>(config); 
            break;
        default:
            JST_FATAL("Backend not supported.");
            throw Result::ERROR;
    }

    return Result::SUCCESS;
}

std::shared_ptr<Window>& Get(const bool& safe = true);

template<class T>
inline Result JETSTREAM_API Create(std::shared_ptr<T>& member, 
                                   const auto& config) {
    auto result = Create(Get()->implementation(), member, config);

    if constexpr (std::is_same<T, Surface>::value) {
        if (result == Result::SUCCESS) {
            return Get()->bind(member);
        }
    }

    return result;
}

template<Device D>
inline Result JETSTREAM_API Initialize(const Window::Config& config) {
    if (Get(false)) {
        JST_FATAL("Render already initialized.");
        throw Result::ERROR;
    }

    return Create(D, Get(false), config);
}

const Result JETSTREAM_API Create();
const Result JETSTREAM_API Destroy();
const Result JETSTREAM_API Begin();
const Result JETSTREAM_API End();
const Result JETSTREAM_API Synchronize();
const bool JETSTREAM_API KeepRunning();

}  // namespace Jetstream::Render

#endif
