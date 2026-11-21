#include "crash_report.hh"

namespace Jetstream {

CrashReportModalPresenter::CrashReportModalPresenter(const PresenterContext& context) : context(context) {}

CrashReportView::Config CrashReportModalPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;

    return CrashReportView::Config{
        .report = context.state.crashReport.preview,
        .count = context.state.crashReport.reports.size(),
        .onSubmit = [enqueue]() {
            enqueue(MailSubmitCrashReport{});
        },
        .onDismiss = [enqueue]() {
            enqueue(MailDismissCrashReport{});
        },
    };
}

}  // namespace Jetstream
