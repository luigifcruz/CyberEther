#include "jetstream/logger.hh"

#include "jetstream/backend/devices/webgpu/base.hh"

namespace Jetstream::Backend {

WebGPU::WebGPU(const Config& _config) : config(_config), cache({}) {
    // Create application.

    device = emscripten_webgpu_get_device();

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
