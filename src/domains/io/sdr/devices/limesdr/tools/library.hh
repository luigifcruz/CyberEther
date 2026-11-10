#ifndef JETSTREAM_DOMAINS_IO_SDR_DEVICES_LIMESDR_TOOLS_LIBRARY_HH
#define JETSTREAM_DOMAINS_IO_SDR_DEVICES_LIMESDR_TOOLS_LIBRARY_HH

#include <atomic>
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <LimeSuite.h>

#include "jetstream/types.hh"

namespace Jetstream::Radio {

// One implementation of the LimeSuite C API. The bundled LimeSuite is linked
// in. LimeSuiteNG, which adds the LimeSDR XTRX and other PCIe boards, is loaded
// at runtime through its legacy C API when it is installed. Both share the same
// declarations except for the antenna name length.
struct LmsApi {
    struct Board {
        std::string name;
        std::string serial;
        std::string address;
    };

    std::vector<Board> (*boards)() = nullptr;
    std::size_t nameLength = sizeof(lms_name_t);

    int (*Open)(lms_device_t**, const lms_info_str_t, void*) = nullptr;
    int (*Close)(lms_device_t*) = nullptr;
    int (*Init)(lms_device_t*) = nullptr;
    int (*EnableChannel)(lms_device_t*, bool, size_t, bool) = nullptr;
    int (*SetSampleRate)(lms_device_t*, float_type, size_t) = nullptr;
    int (*GetSampleRate)(lms_device_t*, bool, size_t, float_type*, float_type*) = nullptr;
    int (*GetSampleRateRange)(lms_device_t*, bool, lms_range_t*) = nullptr;
    int (*SetLOFrequency)(lms_device_t*, bool, size_t, float_type) = nullptr;
    int (*GetLOFrequency)(lms_device_t*, bool, size_t, float_type*) = nullptr;
    int (*GetLOFrequencyRange)(lms_device_t*, bool, lms_range_t*) = nullptr;
    int (*GetAntennaList)(lms_device_t*, bool, size_t, lms_name_t*) = nullptr;
    int (*SetAntenna)(lms_device_t*, bool, size_t, size_t) = nullptr;
    int (*SetGaindB)(lms_device_t*, bool, size_t, unsigned) = nullptr;
    int (*SetLPFBW)(lms_device_t*, bool, size_t, float_type) = nullptr;
    int (*GetLPFBWRange)(lms_device_t*, bool, lms_range_t*) = nullptr;
    int (*Calibrate)(lms_device_t*, bool, size_t, double, unsigned) = nullptr;
    int (*EnableCache)(lms_device_t*, bool) = nullptr;
    int (*SetupStream)(lms_device_t*, lms_stream_t*) = nullptr;
    int (*DestroyStream)(lms_device_t*, lms_stream_t*) = nullptr;
    int (*StartStream)(lms_stream_t*) = nullptr;
    int (*StopStream)(lms_stream_t*) = nullptr;
    int (*RecvStream)(lms_stream_t*, void*, size_t, lms_stream_meta_t*, unsigned) = nullptr;
    int (*GetStreamStatus)(lms_stream_t*, lms_stream_status_t*) = nullptr;
    const char* (*GetLastErrorMessage)() = nullptr;

    std::atomic<U64> openDevices{0};
    std::mutex tuningMutex;

    // Receive antennas as (index, name) pairs.
    std::vector<std::pair<int, std::string>> antennas(lms_device_t* handle) const;

    static LmsApi& Bundled();
    // Returns nullptr when LimeSuiteNG is not installed.
    static LmsApi* Ng();
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_DEVICES_LIMESDR_TOOLS_LIBRARY_HH
