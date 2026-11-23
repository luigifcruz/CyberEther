#include "container.hh"

namespace Jetstream {

ModalPresenter::ModalPresenter(const PresenterContext& context) : context(context),
                                                                  flowgraphInfo(context),
                                                                  renameBlock(context),
                                                                  benchmark(context),
                                                                  dependencies(context),
                                                                  feedback(context),
                                                                  onboarding(context),
                                                                  remoteQr(context),
                                                                  docsSearch(context),
                                                                  crashReport(context),
                                                                  missingPlugins(context),
                                                                  planNotice(context) {}

ModalView::Config ModalPresenter::build() const {
    const auto enqueue = context.callbacks.enqueueMail;

    ModalView::Config config;
    config.content = context.state.modal.content;
    config.onClose = [enqueue,
                      onboarding = config.content == ModalContent::Onboarding,
                      crashReport = config.content == ModalContent::CrashReport]() {
        if (onboarding) {
            enqueue(MailDismissOnboarding{});
        } else if (crashReport) {
            enqueue(MailDismissCrashReport{});
        } else {
            enqueue(MailCloseModal{});
        }
    };

    if (!config.content.has_value()) {
        return config;
    }

    switch (config.content.value()) {
        case ModalContent::About:
            break;
        case ModalContent::FlowgraphInfo:
            config.flowgraphInfo = flowgraphInfo.build();
            break;
        case ModalContent::RenameBlock:
            config.renameBlock = renameBlock.build();
            break;
        case ModalContent::Benchmark:
            config.benchmark = benchmark.build();
            break;
        case ModalContent::Dependencies:
            config.dependencies = dependencies.build();
            break;
        case ModalContent::Feedback:
            config.feedback = feedback.build();
            break;
        case ModalContent::RemoteQr:
            config.remoteQr = remoteQr.build();
            break;
        case ModalContent::DocsSearch:
            config.docsSearch = docsSearch.build();
            break;
        case ModalContent::Onboarding:
            config.onboarding = onboarding.build();
            break;
        case ModalContent::CrashReport:
            config.crashReport = crashReport.build();
            break;
        case ModalContent::MissingPlugins:
            config.missingPlugins = missingPlugins.build();
            break;
        case ModalContent::PlanNotice:
            config.planNotice = planNotice.build();
            break;
    }

    return config;
}

}  // namespace Jetstream
