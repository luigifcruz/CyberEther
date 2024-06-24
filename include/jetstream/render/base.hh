#ifndef JETSTREAM_RENDER_BASE_HH
#define JETSTREAM_RENDER_BASE_HH

#include "jetstream/logger.hh"
#include "jetstream/types.hh"
#include "jetstream/macros.hh"

#include "jetstream/render/macros.hh"
#include "jetstream/render/types.hh"

#include "jetstream/render/base/buffer.hh"
#include "jetstream/render/base/draw.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/render/base/program.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/vertex.hh"

#ifdef JETSTREAM_RENDER_METAL_AVAILABLE
#include "jetstream/render/devices/metal/window.hh"
#include "jetstream/render/devices/metal/surface.hh"
#include "jetstream/render/devices/metal/program.hh"
#include "jetstream/render/devices/metal/kernel.hh"
#include "jetstream/render/devices/metal/buffer.hh"
#include "jetstream/render/devices/metal/draw.hh"
#include "jetstream/render/devices/metal/texture.hh"
#include "jetstream/render/devices/metal/vertex.hh"
#endif

#ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
#include "jetstream/render/devices/vulkan/window.hh"
#include "jetstream/render/devices/vulkan/surface.hh"
#include "jetstream/render/devices/vulkan/program.hh"
#include "jetstream/render/devices/vulkan/kernel.hh"
#include "jetstream/render/devices/vulkan/buffer.hh"
#include "jetstream/render/devices/vulkan/draw.hh"
#include "jetstream/render/devices/vulkan/texture.hh"
#include "jetstream/render/devices/vulkan/vertex.hh"
#endif

#ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
#include "jetstream/render/devices/webgpu/window.hh"
#include "jetstream/render/devices/webgpu/surface.hh"
#include "jetstream/render/devices/webgpu/program.hh"
#include "jetstream/render/devices/webgpu/kernel.hh"
#include "jetstream/render/devices/webgpu/buffer.hh"
#include "jetstream/render/devices/webgpu/draw.hh"
#include "jetstream/render/devices/webgpu/texture.hh"
#include "jetstream/render/devices/webgpu/vertex.hh"
#endif

#endif
