#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CRASH_REPORT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CRASH_REPORT_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../support/crash_report.hh"

#include <optional>
#include <tuple>

namespace Jetstream {

struct CrashReportActions {
    using Filter = std::tuple<MailOpenCrashReport,
                              MailSubmitCrashReport,
                              MailDismissCrashReport>;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    CrashReportActions(DefaultCompositorState& state,
                       DefaultCompositorCallbacks& callbacks);
    static std::optional<MailOpenCrashReport> Detect();
    Result handle(const MailOpenCrashReport& msg);
    Result handle(const MailSubmitCrashReport&);
    Result handle(const MailDismissCrashReport&);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CRASH_REPORT_HH
