#ifndef RENDER_H
#define RENDER_H

#include "render/type.hpp"
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

#ifdef RENDER_METAL_AVAILABLE
#include "render/metal/instance.hpp"
#include "render/metal/program.hpp"
#include "render/metal/surface.hpp"
#include "render/metal/texture.hpp"
#include "render/metal/vertex.hpp"
#endif

namespace Render {

inline std::vector<API> AvailableAPIs = {
#ifdef RENDER_GLES_AVAILABLE
    API::GLES,
#endif
#ifdef RENDER_METAL_AVAILABLE
    API::METAL,
#endif
};

std::shared_ptr<Instance> Instantiate(API api_hint, Instance::Config& cfg, bool force = false);

} // namespace Render

#endif
