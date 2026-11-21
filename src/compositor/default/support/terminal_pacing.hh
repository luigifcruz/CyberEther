#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_TERMINAL_PACING_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_SUPPORT_TERMINAL_PACING_HH

#include <algorithm>
#include <chrono>
#include <optional>

namespace Jetstream {

// The caller supplies time so batching and spacing do not depend on a reader's
// scheduling or on how the PTY combines writes into reads.
class TerminalInputPacer {
 public:
    using Clock = std::chrono::steady_clock;
    static constexpr auto Spacing = std::chrono::milliseconds(8);

    std::chrono::milliseconds delay(Clock::time_point now) const;
    void queue();
    void sent(Clock::time_point now);

 private:
    std::optional<Clock::time_point> lastSent;
    bool queued = false;
};

}  // namespace Jetstream

#endif
