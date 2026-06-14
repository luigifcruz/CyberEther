#ifndef JETSTREAM_BACKEND_DEVICE_CPU_HH
#define JETSTREAM_BACKEND_DEVICE_CPU_HH

#include "jetstream/backend/config.hh"

#include <string>

namespace Jetstream::Backend {

class CPU {
 public:
    explicit CPU(const Config& config);

    const std::string& getPythonRuntimePath() const;

  private:
    Config config;
};

}  // namespace Jetstream::Backend

#endif
