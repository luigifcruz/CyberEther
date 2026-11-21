#include "terminal_pacing.hh"

namespace Jetstream {

std::chrono::milliseconds TerminalInputPacer::delay(Clock::time_point now) const {
    if (queued) {
        return Spacing;
    }
    if (!lastSent) {
        return std::chrono::milliseconds::zero();
    }
    return std::max(std::chrono::milliseconds::zero(),
                    std::chrono::ceil<std::chrono::milliseconds>(*lastSent + Spacing - now));
}

void TerminalInputPacer::queue() { queued = true; }

void TerminalInputPacer::sent(Clock::time_point now) {
    if (queued) {
        lastSent = now;
        queued = false;
    }
}

}  // namespace Jetstream
