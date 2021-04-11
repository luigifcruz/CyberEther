#ifndef RENDER_H
#define RENDER_H

#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>

#include "types.hpp"
#include "magic_enum.hpp"
#include "render_config.hpp"

#ifdef RENDER_GLES_AVAILABLE
#include "gles/api.hpp"
#include "gles/instance.hpp"
#include "gles/state.hpp"
#include "gles/program.hpp"
#include "gles/surface.hpp"
#endif

namespace Render {

inline std::vector<API> AvailableAPIs = {
#ifdef RENDER_GLES_AVAILABLE
    API::GLES,
#endif
};

} // namespace Render

#endif
