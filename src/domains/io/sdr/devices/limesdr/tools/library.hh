#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_LIMESDR_TOOLS_LIBRARY_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_LIMESDR_TOOLS_LIBRARY_HH

#include <atomic>
#include <mutex>
#include <vector>

#include <ConnectionRegistry.h>

#include "jetstream/types.hh"

namespace Jetstream::Radio {

struct LimeSdrLibrary {
    static std::vector<lime::ConnectionHandle> Boards();
    static std::atomic<U64>& OpenDevices();
    static std::mutex& TuningMutex();
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_LIMESDR_TOOLS_LIBRARY_HH
