#ifndef RENDER_H
#define RENDER_H

#include "render/types.hpp"
#include "render_config.hpp"

#include "render/base/instance.hpp"
#include "render/base/program.hpp"
#include "render/base/surface.hpp"
#include "render/base/texture.hpp"

#ifdef RENDER_GLES_AVAILABLE
#include "render/gles/instance.hpp"
#include "render/gles/program.hpp"
#include "render/gles/surface.hpp"
#include "render/gles/texture.hpp"
#endif

namespace Render {

inline std::vector<API> AvailableAPIs = {
#ifdef RENDER_GLES_AVAILABLE
    API::GLES,
#endif
};

} // namespace Render

#endif
