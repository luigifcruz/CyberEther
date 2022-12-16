#ifndef JETSTREAM_BACKEND_DEVICE_METAL_HH
#define JETSTREAM_BACKEND_DEVICE_METAL_HH

#include "jetstream/backend/devices/metal/bindings.hpp"
#include "jetstream/backend/config.hh"

#if __APPLE__
    #include <TargetConditionals.h>
#endif

namespace Jetstream::Backend {

class Metal {
 public:
    explicit Metal(const Config& config);
    ~Metal();

    const std::string getDeviceName() const;
    const bool hasUnifiedMemory() const;
    const bool getLowPowerStatus() const;
    const U64 physicalMemory() const;
    const U64 getActiveProcessorCount() const; 
    const U64 getTotalProcessorCount() const;
    const U64 getThermalState() const;

    constexpr MTL::Device* getDevice() {
        return device;
    }

 private:
    MTL::Device* device;
    NS::ProcessInfo* info;
};

}  // namespace Jetstream::Backend

#endif
