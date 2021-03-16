#ifndef RENDER_CYBERETHER_H
#define RENDER_CYBERETHER_H

#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>

#include "types.hpp"
#include "magic_enum.hpp"
#include "render_config.hpp"

#ifdef RENDER_OPENGL_BACKEND_AVAILABLE
#include "opengl/opengl.hpp"
#endif

namespace Render {

inline std::vector<Backend> AvailableBackends = {
    #ifdef RENDER_OPENGL_BACKEND_AVAILABLE
    Backend::OPENGL,
    #endif
};

inline std::unique_ptr<Render> GetRender(Backend backend_hint, bool force = false) {
    auto a = AvailableBackends;
    Backend render_id = backend_hint;

    if (std::find(a.begin(), a.end(), render_id) == a.end()) {
        if (force) {
            ASSERT_SUCCESS(Result::NO_RENDER_BACKEND_FOUND);
        }

        for (const auto& b : AvailableBackends) {
#ifdef RENDER_DEBUG
            std::cout << "[RENDER] Selected "
                      << magic_enum::enum_name(backend_hint)
                      << " backend not available, switching to "
                      << magic_enum::enum_name(b)
                      << "." << std::endl;
#endif
            render_id = b;
        }
    }

    switch(render_id) {
#ifdef RENDER_OPENGL_BACKEND_AVAILABLE
        case Backend::OPENGL:
            return std::make_unique<OpenGL>();
#endif
        default:
#ifdef RENDER_DEBUG
            std::cerr << "[RENDER] No backend available." << std::endl;
#endif
            ASSERT_SUCCESS(Result::NO_RENDER_BACKEND_FOUND);
    }
}

} // namespace Render

#endif
