#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_PLAN_NOTICE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_PLAN_NOTICE_HH

#include "../context.hh"
#include "../../model/messages.hh"
#include "../../support/agent/chatgpt.hh"

#include "../../views/modal/plan_notice.hh"

namespace Jetstream {

struct PlanNoticeModalPresenter {
    const PresenterContext& context;

    explicit PlanNoticeModalPresenter(const PresenterContext& context);
    PlanNoticeView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_PLAN_NOTICE_HH
