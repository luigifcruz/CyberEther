#ifndef JETSTREAM_BACKEND_DEVICE_WEBGPU_HH
#define JETSTREAM_BACKEND_DEVICE_WEBGPU_HH

#include <set>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include <emscripten.h>
#include <emscripten/html5.h>
#include <webgpu/webgpu_cpp.h>
#include <emscripten/html5_webgpu.h>

#include "jetstream/backend/config.hh"

namespace Jetstream::Backend {

class WebGPU {
 public:
    explicit WebGPU(const Config& config);
    ~WebGPU();

    std::string getDeviceName() const;
    std::string getApiVersion() const;
    PhysicalDeviceType getPhysicalDeviceType() const;
    bool hasUnifiedMemory() const;
    U64 getPhysicalMemory() const;
    U64 getTotalProcessorCount() const;
    bool getLowPowerStatus() const;
    U64 getThermalState() const;

    constexpr wgpu::Device& getDevice() {
        return device;
    }

    constexpr wgpu::Adapter& getAdapter() {
        return adapter;
    }

 private:
    Config config;

    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Surface surface;

    struct {
        std::string deviceName;
        std::string apiVersion;
        PhysicalDeviceType physicalDeviceType;
        bool hasUnifiedMemory;
        U64 physicalMemory;
        U64 totalProcessorCount;
        bool lowPowerStatus;
        U64 getThermalState;
    } cache;
};

}  // namespace Jetstream::Backend

#endif
