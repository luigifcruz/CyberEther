#ifndef JETSTREAM_DOMAINS_IO_SDR_TOOLS_TIMESTAMP_FILTER_HH
#define JETSTREAM_DOMAINS_IO_SDR_TOOLS_TIMESTAMP_FILTER_HH

#include <optional>
#include <string>

#include "../devices/device.hh"

namespace Jetstream::Radio {

class JETSTREAM_API TimestampFilter {
 public:
    struct Decision {
        Device::Status status = Device::Status::Failure;
        U64 skip = 0;
        U64 lost = 0;
        std::string error;
    };

    void restart(U64 guardSamples, U64 maximumStart);
    void retune(U64 validFrom);
    void lose(U64 validFrom);
    Decision accept(U64 timestamp, U64 count);
    U64 pending() const;
    U64 epoch() const;

 private:
    U64 currentEpoch = 0;
    U64 guard = 0;
    U64 maximumStart = 0;
    bool starting = false;
    U64 boundary = 0;
    std::optional<U64> cursor;
    std::optional<U64> next;
    bool gap = false;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_TOOLS_TIMESTAMP_FILTER_HH
