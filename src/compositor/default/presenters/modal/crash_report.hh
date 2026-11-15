#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CRASH_REPORT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CRASH_REPORT_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/crash_report.hh"

namespace Jetstream {

struct CrashReportModalPresenter {
    const PresenterContext& context;

    explicit CrashReportModalPresenter(const PresenterContext& context);
    CrashReportView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CRASH_REPORT_HH
