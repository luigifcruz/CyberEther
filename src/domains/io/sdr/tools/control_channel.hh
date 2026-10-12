#ifndef JETSTREAM_DOMAINS_IO_SDR_TOOLS_CONTROL_CHANNEL_HH
#define JETSTREAM_DOMAINS_IO_SDR_TOOLS_CONTROL_CHANNEL_HH

#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>

#include "jetstream/macros.hh"
#include "jetstream/types.hh"

namespace Jetstream::Radio {

class JETSTREAM_API ControlChannel {
 public:
    using Request = std::function<Result()>;

    void open();
    void close();
    void submit(const Request& request);
    Result post(const Request& request, std::chrono::milliseconds timeout);
    bool pending() const;
    void service();

 private:
    struct Pending {
        Request request;
        std::shared_ptr<std::promise<Result>> done;
    };

    mutable std::mutex mutex;
    std::optional<Pending> latest;
    bool active = false;
};

}  // namespace Jetstream::Radio

#endif  // JETSTREAM_DOMAINS_IO_SDR_TOOLS_CONTROL_CHANNEL_HH
