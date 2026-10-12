#ifndef JETSTREAM_DOMAINS_IO_SDR_TOOLS_CONTROL_CHANNEL_HH
#define JETSTREAM_DOMAINS_IO_SDR_TOOLS_CONTROL_CHANNEL_HH

#include <chrono>
#include <functional>
#include <future>
#include <mutex>
#include <optional>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

// Hands device changes from other threads to the receive thread, which runs
// them between reads. Used by the SDR module.

class JETSTREAM_API ControlChannel {
 public:
    using Request = std::function<Result()>;

    void open();
    void close();
    Result post(const Request& request, std::chrono::milliseconds timeout);
    void service();

 private:
    struct Pending {
        Request request;
        std::promise<Result> done;
    };

    std::mutex mutex;
    std::optional<Pending> pending;
    bool active = false;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_TOOLS_CONTROL_CHANNEL_HH
