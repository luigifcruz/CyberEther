#include "render/base.hpp"
#include "render/tools/magic_enum.hpp"

namespace Render {

static void* __Instance__;
static Backend __Backend__;

void* __GetInstance() {
    return __Instance__;
}

Backend __GetBackend() {
    return __Backend__;
}

Result Init(const Backend& hint,
        const Instance::Config& config, bool forceApi) {
    auto backend = hint;

    if (std::find(AvailableBackends.begin(), AvailableBackends.end(),
                hint) == AvailableBackends.end()) {
        if (forceApi) {
            RENDER_CHECK_THROW(Result::NO_RENDER_BACKEND_FOUND);
        }

        for (const auto& a : AvailableBackends) {
#ifdef RENDER_DEBUG
            std::cout << "[RENDER] Selected "
                      << magic_enum::enum_name(hint)
                      << " backend not available, switching to "
                      << magic_enum::enum_name(a)
                      << "." << std::endl;
#endif
            backend = a;
        }
    }

    switch (backend) {
#ifdef RENDER_GLES_AVAILABLE
        case Backend::GLES:
            __Instance__ = new Broker<GLES>(config);
            break;
#endif
#ifdef RENDER_METAL_AVAILABLE
        case Backend::Metal:
            __Instance__ = new Broker<Metal>(config);
            break;
#endif
        default:
#ifdef RENDER_DEBUG
            std::cerr << "[RENDER] No backend available." << std::endl;
#endif
            return Result::NO_RENDER_BACKEND_FOUND;
    }

    __Backend__ = backend;

    return Result::SUCCESS;
}

Result Create() {
    return Get()->create();
}

Result Destroy() {
    return Get()->destroy();
}

Result Begin() {
    return Get()->begin();
}

Result End() {
    return Get()->end();
}

Result Synchronize() {
    return Get()->synchronize();
}

bool KeepRunning() {
    return Get()->keepRunning();
}

bool HasCudaInterop() {
    return Get()->hasCudaInterop();
}

}  // namespace Render
