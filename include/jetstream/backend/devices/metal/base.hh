#ifndef JETSTREAM_BACKEND_DEVICE_METAL_HH
#define JETSTREAM_BACKEND_DEVICE_METAL_HH

#pragma GCC diagnostic push 
#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
#pragma GCC diagnostic ignored "-Wnested-anon-types"
#pragma GCC diagnostic ignored "-Wdtor-name"
#include "jetstream/backend/devices/metal/bindings.hpp"
#pragma GCC diagnostic pop
#include "jetstream/backend/config.hh"

namespace Jetstream::Backend {

class Metal {
 public:
    explicit Metal(const Config& config);
    ~Metal();

    bool isAvailable() const;
    std::string getDeviceName() const;
    std::string getApiVersion() const;
    bool hasUnifiedMemory() const;
    bool getLowPowerStatus() const;
    U64 getPhysicalMemory() const;
    U64 getActiveProcessorCount() const; 
    U64 getTotalProcessorCount() const;
    U64 getThermalState() const;
    PhysicalDeviceType getPhysicalDeviceType() const;

    constexpr const U64& getDeviceId() const {
        return config.deviceId;
    }

    constexpr MTL::Device* getDevice() {
        return device;
    }

 private:
    Config config;
    MTL::Device* device;
    NS::ProcessInfo* info;
    bool _isAvailable = false;
};

}  // namespace Jetstream::Backend

#endif
