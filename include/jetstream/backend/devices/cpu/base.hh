#ifndef JETSTREAM_BACKEND_DEVICE_CPU_HH
#define JETSTREAM_BACKEND_DEVICE_CPU_HH

#include "jetstream/backend/config.hh"

namespace Jetstream::Backend {

class CPU {
 public:
    explicit CPU(const Config& config);
};

}  // namespace Jetstream::Backend

#endif
