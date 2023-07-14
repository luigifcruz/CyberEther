#ifndef JETSTREAM_BACKEND_DEVICE_WEBGPU_HELPERS_HH
#define JETSTREAM_BACKEND_DEVICE_WEBGPU_HELPERS_HH

#include <span>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"

namespace Jetstream::Backend {

inline wgpu::ShaderModule LoadShader(const std::span<const U8>& data, wgpu::Device& device) {
    wgpu::ShaderModuleWGSLDescriptor wgsl{};
	  wgsl.sType = wgpu::SType::ShaderModuleWGSLDescriptor;
	  wgsl.source = reinterpret_cast<const char*>(data.data());

	  wgpu::ShaderModuleDescriptor desc{};
	  desc.nextInChain = reinterpret_cast<wgpu::ChainedStruct*>(&wgsl);

	  return device.CreateShaderModule(&desc);
}

}  // namespace Jetstream::Backend

#endif