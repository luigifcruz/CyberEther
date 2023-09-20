#ifndef JETSTREAM_BACKEND_DEVICE_METAL_HH
#define JETSTREAM_BACKEND_DEVICE_METAL_HH

#include "jetstream/backend/devices/metal/bindings.hpp"
#include "jetstream/backend/config.hh"

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

namespace Jetstream::Backend {

class Metal {
 public:
    explicit Metal(const Config& config);
    ~Metal();

    std::string getDeviceName() const;
    std::string getApiVersion() const;
    bool hasUnifiedMemory() const;
    bool getLowPowerStatus() const;
    U64 getPhysicalMemory() const;
    U64 getActiveProcessorCount() const; 
    U64 getTotalProcessorCount() const;
    U64 getThermalState() const;
    PhysicalDeviceType getPhysicalDeviceType() const;

    constexpr MTL::Device* getDevice() {
        return device;
    }

 private:
    MTL::Device* device;
    NS::ProcessInfo* info;
};

}  // namespace Jetstream::Backend

#endif
