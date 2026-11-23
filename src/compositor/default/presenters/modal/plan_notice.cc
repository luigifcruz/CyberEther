#include "plan_notice.hh"

namespace Jetstream {

PlanNoticeModalPresenter::PlanNoticeModalPresenter(const PresenterContext& context) : context(context) {}

PlanNoticeView::Config PlanNoticeModalPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;
    return {
        .onManageUsage = [enqueue]() {
            enqueue(MailOpenUrl{.url = AgentChatGpt::UsageUrl, .notifyResult = true});
        },
        .onAddModels = [enqueue]() {
            enqueue(MailCloseModal{});
            enqueue(MailOpenAgentModelPicker{});
        },
    };
}

}  // namespace Jetstream
