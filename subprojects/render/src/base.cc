#include "render/base.hpp"
#include "render/tools/magic_enum.hpp"

namespace Render {

std::shared_ptr<Instance> Instantiate(API api_hint, Instance::Config& cfg, bool force) {
    auto api = api_hint;

    if (std::find(AvailableAPIs.begin(), AvailableAPIs.end(),
                api_hint) == AvailableAPIs.end()) {
        if (force) {
            RENDER_CHECK_THROW(Result::NO_RENDER_BACKEND_FOUND);
        }

        for (const auto& a : AvailableAPIs) {
#ifdef RENDER_DEBUG
            std::cout << "[RENDER] Selected "
                      << magic_enum::enum_name(api_hint)
                      << " API not available, switching to "
                      << magic_enum::enum_name(a)
                      << "." << std::endl;
#endif
            api = a;
        }
    }

    switch (api) {
#ifdef RENDER_GLES_AVAILABLE
        case API::GLES:
            return std::make_shared<GLES>(cfg);
#endif
#ifdef RENDER_METAL_AVAILABLE
        case API::METAL:
            return std::make_shared<Metal>(cfg);
#endif
        default:
#ifdef RENDER_DEBUG
            std::cerr << "[RENDER] No API available." << std::endl;
#endif
            RENDER_CHECK_THROW(Result::NO_RENDER_BACKEND_FOUND);
    }
}

} // namespace Render
