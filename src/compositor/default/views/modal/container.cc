#include "container.hh"

namespace Jetstream {

void ModalView::update(Config config) {
    this->config = std::move(config);

    if (previousContent == ModalContent::RenameBlock &&
        this->config.content != ModalContent::RenameBlock) {
        renameBlockView.update({});
    }
    if (previousContent == ModalContent::Feedback &&
        this->config.content != ModalContent::Feedback) {
        feedbackView.update({});
    }

    if (!this->config.content.has_value()) {
        previousContent = this->config.content;
        return;
    }

    switch (this->config.content.value()) {
        case ModalContent::About:
            aboutView.update({});
            break;
        case ModalContent::FlowgraphInfo:
            if (this->config.flowgraphInfo.has_value()) {
                auto viewConfig = this->config.flowgraphInfo.value();
                flowgraphInfoView.update(std::move(viewConfig));
            }
            break;
        case ModalContent::RenameBlock:
            if (this->config.renameBlock.has_value()) {
                auto viewConfig = this->config.renameBlock.value();
                renameBlockView.update(std::move(viewConfig));
            }
            break;
        case ModalContent::Benchmark: {
            auto viewConfig = this->config.benchmark;
            benchmarkView.update(std::move(viewConfig));
            break;
        }
        case ModalContent::Dependencies: {
            auto viewConfig = this->config.dependencies;
            dependencyReviewView.update(std::move(viewConfig));
            break;
        }
        case ModalContent::Feedback: {
            auto viewConfig = this->config.feedback;
            feedbackView.update(std::move(viewConfig));
            break;
        }
        case ModalContent::RemoteQr: {
            auto viewConfig = this->config.remoteQr;
            remoteQrView.update(std::move(viewConfig));
            break;
        }
        case ModalContent::DocsSearch: {
            auto viewConfig = this->config.docsSearch;
            docsSearchView.update(std::move(viewConfig));
            break;
        }
        case ModalContent::Onboarding: {
            auto viewConfig = this->config.onboarding;
            onboardingView.update(std::move(viewConfig));
            break;
        }
        case ModalContent::CrashReport: {
            auto viewConfig = this->config.crashReport;
            crashReportView.update(std::move(viewConfig));
            break;
        }
        case ModalContent::MissingPlugins: {
            auto viewConfig = this->config.missingPlugins;
            missingPluginsView.update(std::move(viewConfig));
            break;
        }
        case ModalContent::PlanNotice: {
            auto viewConfig = this->config.planNotice;
            planNoticeView.update(std::move(viewConfig));
            break;
        }
    }

    previousContent = this->config.content;
}

Sakura::ModalLayer::Config ModalView::layer() {
    if (config.content == ModalContent::DocsSearch) {
        return docsSearchView.layer();
    }
    return {
        .content = content(),
        .size = modalSize(),
        .closeButton = config.content != ModalContent::Feedback && config.content != ModalContent::RemoteQr &&
                       config.content != ModalContent::Onboarding && config.content != ModalContent::FlowgraphInfo &&
                       config.content != ModalContent::CrashReport &&
                       config.content != ModalContent::MissingPlugins &&
                       config.content != ModalContent::PlanNotice,
        .onClose = [this]() {
            if (config.onClose) {
                config.onClose();
            }
        },
    };
}

Sakura::Component* ModalView::content() {
    if (!config.content.has_value()) {
        return nullptr;
    }
    switch (config.content.value()) {
        case ModalContent::About:
            return &aboutView.component();
        case ModalContent::FlowgraphInfo:
            return config.flowgraphInfo.has_value() ? &flowgraphInfoView.component() : nullptr;
        case ModalContent::RenameBlock:
            return config.renameBlock.has_value() ? &renameBlockView.component() : nullptr;
        case ModalContent::Benchmark:
            return &benchmarkView.component();
        case ModalContent::Dependencies:
            return &dependencyReviewView.component();
        case ModalContent::Feedback:
            return &feedbackView.component();
        case ModalContent::RemoteQr:
            return &remoteQrView.component();
        case ModalContent::DocsSearch:
            return nullptr;
        case ModalContent::Onboarding:
            return &onboardingView.component();
        case ModalContent::CrashReport:
            return &crashReportView.component();
        case ModalContent::MissingPlugins:
            return &missingPluginsView.component();
        case ModalContent::PlanNotice:
            return &planNoticeView.component();
    }
    return nullptr;
}

Extent2D<F32> ModalView::modalSize() const {
    if (config.content == ModalContent::Feedback) {
        return Extent2D<F32>{520.0f, 0.0f};
    }
    if (config.content == ModalContent::FlowgraphInfo) {
        return Extent2D<F32>{FlowgraphInfoView::Width, 0.0f};
    }
    if (config.content == ModalContent::RemoteQr) {
        return Extent2D<F32>{460.0f, 0.0f};
    }
    if (config.content == ModalContent::Dependencies) {
        return Extent2D<F32>{760.0f, 0.0f};
    }
    if (config.content == ModalContent::Onboarding) {
        return OnboardingView::Size();
    }
    if (config.content == ModalContent::CrashReport) {
        return Extent2D<F32>{CrashReportView::Width, 0.0f};
    }
    if (config.content == ModalContent::MissingPlugins) {
        return Extent2D<F32>{MissingPluginsView::Width, 0.0f};
    }
    if (config.content == ModalContent::PlanNotice) {
        return Extent2D<F32>{PlanNoticeView::Width, 0.0f};
    }
    return Sakura::ModalLayer::Config{}.size;
}

}  // namespace Jetstream
