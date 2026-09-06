#ifndef JETSTREAM_DOMAINS_IO_SOAPY_RECEIVE_STATUS_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_RECEIVE_STATUS_HH

#include <chrono>

#include <SoapySDR/Errors.hpp>

#include <jetstream/types.hh>

namespace Jetstream::detail {

struct SoapyReceiveStatus {
    using Clock = std::chrono::steady_clock;

    enum class Action {
        Samples,
        Retry,
        WarnOverflow,
        Fail,
    };

    Action handle(const int result, const Clock::time_point now = Clock::now()) {
        if (result > 0) {
            return Action::Samples;
        }
        if (result == 0 || result == SOAPY_SDR_TIMEOUT) {
            return Action::Retry;
        }
        if (result != SOAPY_SDR_OVERFLOW) {
            return Action::Fail;
        }

        ++deviceOverflows;
        if (deviceOverflows == 1 || now >= nextWarning) {
            nextWarning = now + std::chrono::seconds(1);
            return Action::WarnOverflow;
        }
        return Action::Retry;
    }

    U64 deviceOverflows = 0;
    Clock::time_point nextWarning{};
};

}  // namespace Jetstream::detail

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_RECEIVE_STATUS_HH
