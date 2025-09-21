#ifndef JETSTREAM_BACKEND_DEVICE_WEBGPU_HELPERS_HH
#define JETSTREAM_BACKEND_DEVICE_WEBGPU_HELPERS_HH

#include <vector>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"

namespace Jetstream::Backend {

inline WGPUShaderModule LoadShader(const std::vector<U8>& data, WGPUDevice device) {
    WGPUShaderSourceWGSL wgsl = {};
    wgsl.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl.code = { (const char*)data.data(), data.size() };

    WGPUShaderModuleDescriptor desc = {};
    desc.nextInChain = &wgsl.chain;

    return wgpuDeviceCreateShaderModule(device, &desc);
}

}  // namespace Jetstream::Backend

#endif
