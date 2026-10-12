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
    if (latest && latest->done) {
        latest->done->set_value(Result::ERROR);
    }
    latest.reset();
}

void ControlChannel::submit(const Request& request) {
    {
        std::lock_guard lock(mutex);
        if (active) {
            if (latest && latest->done) {
                latest->done->set_value(Result::ERROR);
            }
            latest = Pending{request, nullptr};
            return;
        }
    }
    request();
}

Result ControlChannel::post(const Request& request, const std::chrono::milliseconds timeout) {
    auto done = std::make_shared<std::promise<Result>>();
    auto result = done->get_future();
    {
        std::lock_guard lock(mutex);
        if (!active) {
            return request();
        }
        if (latest && latest->done) {
            latest->done->set_value(Result::ERROR);
        }
        latest = Pending{request, done};
    }
    if (result.wait_for(timeout) != std::future_status::ready) {
        std::lock_guard lock(mutex);
        if (latest && latest->done == done) {
            latest.reset();
            JST_ERROR("[SDR] Timed out waiting for the receive thread to apply the change.");
            return Result::ERROR;
        }
    }
    return result.get();
}

bool ControlChannel::pending() const {
    std::lock_guard lock(mutex);
    return latest.has_value();
}

void ControlChannel::service() {
    std::optional<Pending> current;
    {
        std::lock_guard lock(mutex);
        current.swap(latest);
    }
    if (!current) {
        return;
    }
    Result result = Result::ERROR;
    try {
        result = current->request();
    } catch (...) {
        JST_ERROR("[SDR] A device change failed.");
    }
    if (current->done) {
        current->done->set_value(result);
    } else if (result != Result::SUCCESS) {
        JST_WARN("[SDR] A device change did not apply cleanly.");
    }
}

}  // namespace Jetstream::Radio
