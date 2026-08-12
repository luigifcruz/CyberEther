#ifndef JETSTREAM_BACKEND_DEVICE_WEBGPU_HH
#define JETSTREAM_BACKEND_DEVICE_WEBGPU_HH

#include <set>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>

#include <emscripten.h>
#include <webgpu/webgpu.h>

#include "jetstream/backend/config.hh"

namespace Jetstream::Backend {

class WebGPU {
 public:
    explicit WebGPU(const Config& config);
    ~WebGPU();

    static Result InitializeAsync(std::function<void(Result)> completion);
    static void CancelInitialization();

    std::string getDeviceName() const;
    std::string getApiVersion() const;
    PhysicalDeviceType getPhysicalDeviceType() const;
    bool hasUnifiedMemory() const;
    U64 getPhysicalMemory() const;
    U64 getTotalProcessorCount() const;
    bool getLowPowerStatus() const;
    U64 getThermalState() const;

    WGPUDevice getDevice() const {
        return device;
    }

    WGPUAdapter getAdapter() const {
        return adapter;
    }

    WGPUInstance getInstance() const {
        return instance;
    }

 private:
    Config config;

    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;

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
