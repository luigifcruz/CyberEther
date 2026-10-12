#include "control_channel.hh"

#include "jetstream/logger.hh"

namespace Jetstream::Radio {

void ControlChannel::open() {
    std::lock_guard lock(mutex);
    active = true;
}

void ControlChannel::close() {
    std::lock_guard lock(mutex);
    active = false;
    if (pending) {
        pending->done.set_value(Result::ERROR);
        pending.reset();
    }
}

Result ControlChannel::post(const Request& request, const std::chrono::milliseconds timeout) {
    std::future<Result> result;
    {
        std::lock_guard lock(mutex);
        if (!active) {
            return request();
        }
        if (pending) {
            JST_ERROR("[SDR] Another device change is still pending.");
            return Result::ERROR;
        }
        pending.emplace();
        pending->request = request;
        result = pending->done.get_future();
    }
    if (result.wait_for(timeout) != std::future_status::ready) {
        std::lock_guard lock(mutex);
        if (pending) {
            pending.reset();
            JST_ERROR("[SDR] Timed out waiting for the receive thread to apply the change.");
            return Result::ERROR;
        }
    }
    return result.get();
}

void ControlChannel::service() {
    std::optional<Pending> current;
    {
        std::lock_guard lock(mutex);
        if (!pending) {
            return;
        }
        current.swap(pending);
    }
    Result result = Result::ERROR;
    try {
        result = current->request();
    } catch (...) {
        result = Result::ERROR;
    }
    current->done.set_value(result);
}

}  // namespace Jetstream::Radio
