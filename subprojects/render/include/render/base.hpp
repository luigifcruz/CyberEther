#ifndef RENDER_H
#define RENDER_H

#include <vector>
#include <memory>

#include "render/type.hpp"
#include "render_config.hpp"

#include "render/base/instance.hpp"
#include "render/base/program.hpp"
#include "render/base/surface.hpp"
#include "render/base/texture.hpp"
#include "render/base/vertex.hpp"
#include "render/base/draw.hpp"

#ifdef RENDER_GLES_AVAILABLE
#include "render/gles/instance.hpp"
#include "render/gles/program.hpp"
#include "render/gles/surface.hpp"
#include "render/gles/texture.hpp"
#include "render/gles/vertex.hpp"
#include "render/gles/draw.hpp"
#endif

#ifdef RENDER_METAL_AVAILABLE
#include "render/metal/instance.hpp"
#include "render/metal/program.hpp"
#include "render/metal/surface.hpp"
#include "render/metal/texture.hpp"
#include "render/metal/vertex.hpp"
#include "render/metal/draw.hpp"
#endif

namespace Render {

void* __GetInstance();
Backend __GetBackend();

inline std::vector<Backend> AvailableBackends = {
#ifdef RENDER_GLES_AVAILABLE
    Backend::GLES,
#endif
#ifdef RENDER_METAL_AVAILABLE
    Backend::Metal,
#endif
};

template<typename T>
class Broker : public T {
 public:
    using T::T;

    std::shared_ptr<Vertex> newMember(const Render::Vertex::Config& config) {
        return std::make_shared<typename T::Vertex>(config, *this);
    }

    std::shared_ptr<Draw> newMember(const Render::Draw::Config& config) {
        return std::make_shared<typename T::Draw>(config, *this);
    }

    std::shared_ptr<Program> newMember(const Render::Program::Config& config) {
        return std::make_shared<typename T::Program>(config, *this);
    }

    std::shared_ptr<Surface> newMember(const Render::Surface::Config& config) {
        auto surface = std::make_shared<typename T::Surface>(config, *this);
        this->surfaces.push_back(surface);
        return surface;
    }

    std::shared_ptr<Texture> newMember(const Render::Texture::Config& config) {
        return std::make_shared<typename T::Texture>(config, *this);
    }
};

inline auto* Get() {
    if (__GetInstance() == nullptr) {
        std::cerr << "[RENDER] Error. The backend was not initialized." << std::endl;
        RENDER_CHECK_THROW(Result::NO_RENDER_BACKEND_FOUND);
    }

    switch (__GetBackend()) {
#ifdef RENDER_GLES_AVAILABLE
        case Backend::GLES:
            return static_cast<Broker<GLES>*>(__GetInstance());
#endif
#ifdef RENDER_METAL_AVAILABLE
        case Backend::Metal:
            return static_cast<Broker<Metal>*>(__GetInstance());
#endif
        default:
            std::cerr << "[RENDER] No Backend available." << std::endl;
            RENDER_CHECK_THROW(Result::NO_RENDER_BACKEND_FOUND);
    }
}

Result Init(const Backend& backend,
        const Instance::Config& config, bool forceApi = false);

template<typename T>
inline auto Create(const T& config) {
    return Get()->newMember(config);
}

inline auto CreateAndBind(const Surface::Config& config) {
    return Create(config);
}

Result Create();
Result Destroy();
Result Begin();
Result End();
Result Synchronize();
bool KeepRunning();
bool HasCudaInterop();

}  // namespace Render

#endif
