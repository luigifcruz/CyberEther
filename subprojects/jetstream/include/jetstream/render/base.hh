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

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
#include "jetstream/render/vulkan/window.hh"
#include "jetstream/render/vulkan/surface.hh"
#include "jetstream/render/vulkan/program.hh"
#include "jetstream/render/vulkan/buffer.hh"
#include "jetstream/render/vulkan/draw.hh"
#include "jetstream/render/vulkan/texture.hh"
#include "jetstream/render/vulkan/vertex.hh"
#endif

#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
#include "jetstream/render/webgpu/window.hh"
#include "jetstream/render/webgpu/surface.hh"
#include "jetstream/render/webgpu/program.hh"
#include "jetstream/render/webgpu/buffer.hh"
#include "jetstream/render/webgpu/draw.hh"
#include "jetstream/render/webgpu/texture.hh"
#include "jetstream/render/webgpu/vertex.hh"
#endif

#endif
