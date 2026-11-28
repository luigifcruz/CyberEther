#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CRASH_REPORT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CRASH_REPORT_HH

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../support/crash_report.hh"

#include <optional>

namespace Jetstream {

struct CrashReportActions {
    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    CrashReportActions(DefaultCompositorState& state,
                       DefaultCompositorCallbacks& callbacks);
    static std::optional<MailOpenCrashReport> Detect(bool officialPackage);
    Result handle(const MailOpenCrashReport& msg);
    Result handle(const MailSubmitCrashReport&);
    Result handle(const MailDismissCrashReport&);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_CRASH_REPORT_HH
