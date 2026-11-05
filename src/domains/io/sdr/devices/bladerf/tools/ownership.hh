#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_BLADERF_TOOLS_OWNERSHIP_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_BLADERF_TOOLS_OWNERSHIP_HH

#include <string>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

// Holds exclusive ownership until the associated hardware handle has closed.
class JETSTREAM_API BladeRfOwnership {
 public:
    BladeRfOwnership() = default;
    ~BladeRfOwnership();
    BladeRfOwnership(const BladeRfOwnership&) = delete;
    BladeRfOwnership& operator=(const BladeRfOwnership&) = delete;

    Result acquire(const std::string& serial);

 private:
    std::string serial;
};

}  // namespace Jetstream::Radio

#endif
