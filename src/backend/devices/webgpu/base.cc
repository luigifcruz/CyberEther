#include "jetstream/logger.hh"

#include "jetstream/backend/devices/webgpu/base.hh"

namespace Jetstream::Backend {

static void WebGPUErrorCallback(WGPUErrorType error_type, const char* message, void*) {
    const char* error_type_lbl = "";
    switch (error_type) {
        case WGPUErrorType_Validation:  error_type_lbl = "Validation";    break;
        case WGPUErrorType_OutOfMemory: error_type_lbl = "Out of memory"; break;
        case WGPUErrorType_Unknown:     error_type_lbl = "Unknown";       break;
        case WGPUErrorType_DeviceLost:  error_type_lbl = "Device lost";   break;
        default:                        error_type_lbl = "Unknown";
    }
    JST_FATAL("[WebGPU] {} error: {}", error_type_lbl, message);
    JST_CHECK_THROW(Result::FATAL);
}

EM_JS(WGPUAdapter, wgpuInstanceRequestAdapterSync, (), {
    return Module["preinitializedWebGPUAdapter"];
});

WebGPU::WebGPU(const Config& _config) : config(_config), cache({}) {
    // Create application.

    adapter = wgpu::Adapter::Acquire(wgpuInstanceRequestAdapterSync());
    device = wgpu::Device::Acquire(emscripten_webgpu_get_device());

    device.SetUncapturedErrorCallback(&WebGPUErrorCallback, nullptr);

    // Print device information.

    JST_WARN("Due to current Emscripten limitations the device values are inaccurate.");
    JST_INFO("-----------------------------------------------------");
    JST_INFO("Jetstream Heterogeneous Backend [WebGPU]")
    JST_INFO("-----------------------------------------------------");
    JST_INFO("Device Name:     {}", getDeviceName());
    JST_INFO("Device Type:     {}", getPhysicalDeviceType());
    JST_INFO("API Version:     {}", getApiVersion());
    JST_INFO("Unified Memory:  {}", hasUnifiedMemory() ? "YES" : "NO");
    JST_INFO("Processor Count: {}", getTotalProcessorCount());
    JST_INFO("Device Memory:   {:.2f} GB", static_cast<F32>(getPhysicalMemory()) / (1024*1024*1024));
    JST_INFO("Staging Buffer:  {:.2f} MB", static_cast<F32>(config.stagingBufferSize) / JST_MB);
    JST_INFO("-----------------------------------------------------");
}

std::string WebGPU::getDeviceName() const {
    return cache.deviceName;
}

std::string WebGPU::getApiVersion() const {
    return cache.apiVersion;       
}

PhysicalDeviceType WebGPU::getPhysicalDeviceType() const {
    return cache.physicalDeviceType; 
}

bool WebGPU::hasUnifiedMemory() const {
    return cache.hasUnifiedMemory;
}

U64 WebGPU::getPhysicalMemory() const {
    return cache.physicalMemory;
}

U64 WebGPU::getTotalProcessorCount() const {
    return cache.totalProcessorCount;
}

bool WebGPU::getLowPowerStatus() const {
    // TODO: Pool power status periodically.
    return cache.lowPowerStatus;
}

U64 WebGPU::getThermalState() const {
    // TODO: Pool thermal state periodically.
    return cache.getThermalState;
}

}  // namespace Jetstream::Backend
