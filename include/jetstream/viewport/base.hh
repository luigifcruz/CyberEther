#ifndef JETSTREAM_VIEWPORT_BASE_HH
#define JETSTREAM_VIEWPORT_BASE_HH

#ifdef JETSTREAM_VIEWPORT_HEADLESS_AVAILABLE
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/viewport/platforms/headless/vulkan.hh"
#endif
#endif

#ifdef JETSTREAM_VIEWPORT_GLFW_AVAILABLE
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/viewport/platforms/glfw/metal.hh"
#endif
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
#include "jetstream/viewport/platforms/glfw/vulkan.hh"
#endif
#ifdef JETSTREAM_BACKEND_WEBGPU_AVAILABLE
#include "jetstream/viewport/platforms/glfw/webgpu.hh"
#endif
#endif

#ifdef JETSTREAM_VIEWPORT_IOS_AVAILABLE
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/viewport/platforms/ios/metal.hh"
#endif
#endif

#endif
