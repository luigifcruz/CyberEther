#ifndef RENDER_H
#define RENDER_H

#include "render/types.hpp"
#include "render_config.hpp"

#include "render/base/instance.hpp"
#include "render/base/program.hpp"
#include "render/base/surface.hpp"
#include "render/base/texture.hpp"
#include "render/base/vertex.hpp"

#ifdef RENDER_GLES_AVAILABLE
#include "render/gles/instance.hpp"
#include "render/gles/program.hpp"
#include "render/gles/surface.hpp"
#include "render/gles/texture.hpp"
#include "render/gles/vertex.hpp"
#endif

namespace Render {

inline std::vector<API> AvailableAPIs = {
#ifdef RENDER_GLES_AVAILABLE
    API::GLES,
#endif
};

inline std::shared_ptr<Instance> Instantiate(API api_hint, Instance::Config& cfg, bool force = false) {
    auto api = api_hint;

    if (std::find(AvailableAPIs.begin(), AvailableAPIs.end(),
                api_hint) == AvailableAPIs.end()) {
        if (force) {
            RENDER_ASSERT_SUCCESS(Result::NO_RENDER_BACKEND_FOUND);
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
        default:
#ifdef RENDER_DEBUG
            std::cerr << "[RENDER] No API available." << std::endl;
#endif
            RENDER_ASSERT_SUCCESS(Result::NO_RENDER_BACKEND_FOUND);
    }
}

} // namespace Render

#endif
