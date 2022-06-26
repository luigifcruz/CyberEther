#ifndef JETSTREAM_BACKEND_DEVICE_METAL_HH
#define JETSTREAM_BACKEND_DEVICE_METAL_HH

#include "jetstream/backend/devices/metal/bindings.hpp"
#include "jetstream/backend/config.hh"

namespace Jetstream::Backend {

class Metal {
 public:
    explicit Metal(const Config& config);

    constexpr MTL::Device* getDevice() {
      return device;
    }

 private:
    MTL::Device* device;
};

}  // namespace Jetstream::Backend

#endif
