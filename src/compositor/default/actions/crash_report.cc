#include "crash_report.hh"

namespace Jetstream {

CrashReportActions::CrashReportActions(DefaultCompositorState& state,
                                       DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

std::optional<MailOpenCrashReport> CrashReportActions::Detect() {
    auto reports = CrashReport::Pending();
    if (reports.empty()) {
        return std::nullopt;
    }

    std::string preview = CrashReport::Read(reports.back());
    return MailOpenCrashReport{.reports = std::move(reports), .preview = std::move(preview)};
}

Result CrashReportActions::handle(const MailOpenCrashReport& msg) {
    state.crashReport = {.reports = msg.reports, .preview = msg.preview};
    state.modal = {.content = ModalContent::CrashReport};
    return Result::SUCCESS;
}

Result CrashReportActions::handle(const MailSubmitCrashReport&) {
    if (state.crashReport.reports.empty() || state.crashReport.submitting) {
        return Result::SUCCESS;
    }

    state.crashReport.submitting = true;
    callbacks.submitCrashReports(state.crashReport.reports);
    if (state.modal.content == ModalContent::CrashReport) {
        state.modal = {};
    }
    return Result::SUCCESS;
}

Result CrashReportActions::handle(const MailDismissCrashReport&) {
    if (!state.crashReport.submitting) {
        CrashReport::Discard(state.crashReport.reports);
        state.crashReport = {};
    }
    if (state.modal.content == ModalContent::CrashReport) {
        state.modal = {};
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream
