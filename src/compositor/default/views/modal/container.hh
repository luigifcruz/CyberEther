#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH

#include "about.hh"
#include "benchmark.hh"
#include "dependencies.hh"
#include "../flowgraph/modals/close.hh"
#include "../flowgraph/modals/examples.hh"
#include "../flowgraph/modals/info.hh"
#include "../flowgraph/modals/rename.hh"
#include "feedback.hh"
#include "intelligence_model.hh"
#include "intelligence_provider.hh"
#include "settings/base.hh"
#include "plugin.hh"
#include "../../model/ui.hh"
#include "remote.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <optional>
#include <utility>

namespace Jetstream {

struct ModalView {
    struct Config {
        std::optional<ModalContent> content;
        FlowgraphExamplesView::Config flowgraphExamples;
        std::optional<FlowgraphInfoView::Config> flowgraphInfo;
        SettingsView::Config appSettings;
        std::optional<FlowgraphCloseView::Config> flowgraphClose;
        std::optional<RenameBlockView::Config> renameBlock;
        BenchmarkView::Config benchmark;
        RemoteView::Config remoteStreaming;
        PluginView::Config plugin;
        IntelligenceProviderView::Config intelligenceProvider;
        IntelligenceModelView::Config intelligenceModel;
        DependencyReviewView::Config dependencies;
        FeedbackView::Config feedback;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);

        if (previousContent == ModalContent::RenameBlock &&
            this->config.content != ModalContent::RenameBlock) {
            renameBlockView.update({});
        }

        if (!this->config.content.has_value()) {
            previousContent = this->config.content;
            return;
        }

        switch (this->config.content.value()) {
            case ModalContent::About:
                aboutView.update({});
                break;
            case ModalContent::FlowgraphExamples: {
                auto viewConfig = this->config.flowgraphExamples;
                flowgraphExamplesView.update(std::move(viewConfig));
                break;
            }
            case ModalContent::FlowgraphInfo:
                if (this->config.flowgraphInfo.has_value()) {
                    auto viewConfig = this->config.flowgraphInfo.value();
                    flowgraphInfoView.update(std::move(viewConfig));
                }
                break;
            case ModalContent::Settings: {
                auto viewConfig = this->config.appSettings;
                appSettingsView.update(std::move(viewConfig));
                break;
            }
            case ModalContent::FlowgraphClose:
                if (this->config.flowgraphClose.has_value()) {
                    auto viewConfig = this->config.flowgraphClose.value();
                    flowgraphCloseView.update(std::move(viewConfig));
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
            case ModalContent::RemoteStreaming: {
                auto viewConfig = this->config.remoteStreaming;
                remoteView.update(std::move(viewConfig));
                break;
            }
            case ModalContent::Plugin: {
                auto viewConfig = this->config.plugin;
                pluginView.update(std::move(viewConfig));
                break;
            }
            case ModalContent::IntelligenceProvider: {
                auto viewConfig = this->config.intelligenceProvider;
                intelligenceProviderView.update(std::move(viewConfig));
                break;
            }
            case ModalContent::IntelligenceModel: {
                auto viewConfig = this->config.intelligenceModel;
                intelligenceModelView.update(std::move(viewConfig));
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
        }

        previousContent = this->config.content;
    }

    Sakura::ModalLayer::Config layer() {
        return {
            .content = content(),
            .size = modalSize(),
            .onClose = [this]() {
                if (config.onClose) {
                    config.onClose();
                }
            },
        };
    }

 private:
    Sakura::Component* content() {
        if (!config.content.has_value()) {
            return nullptr;
        }
        switch (config.content.value()) {
            case ModalContent::About:
                return &aboutView.component();
            case ModalContent::FlowgraphExamples:
                return &flowgraphExamplesView.component();
            case ModalContent::FlowgraphInfo:
                return config.flowgraphInfo.has_value() ? &flowgraphInfoView.component() : nullptr;
            case ModalContent::Settings:
                return &appSettingsView.component();
            case ModalContent::FlowgraphClose:
                return config.flowgraphClose.has_value() ? &flowgraphCloseView.component() : nullptr;
            case ModalContent::RenameBlock:
                return config.renameBlock.has_value() ? &renameBlockView.component() : nullptr;
            case ModalContent::Benchmark:
                return &benchmarkView.component();
            case ModalContent::RemoteStreaming:
                return &remoteView.component();
            case ModalContent::Plugin:
                return &pluginView.component();
            case ModalContent::IntelligenceProvider:
                return &intelligenceProviderView.component();
            case ModalContent::IntelligenceModel:
                return &intelligenceModelView.component();
            case ModalContent::Dependencies:
                return &dependencyReviewView.component();
            case ModalContent::Feedback:
                return &feedbackView.component();
        }
        return nullptr;
    }

    Extent2D<F32> modalSize() const {
        if (config.content == ModalContent::Settings) {
            return Extent2D<F32>{880.0f, 700.0f};
        }
        if (config.content == ModalContent::Plugin) {
            return Extent2D<F32>{620.0f, 0.0f};
        }
        if (config.content == ModalContent::IntelligenceProvider) {
            return Extent2D<F32>{620.0f, 0.0f};
        }
        if (config.content == ModalContent::IntelligenceModel) {
            return Extent2D<F32>{680.0f, 0.0f};
        }
        if (config.content == ModalContent::Feedback) {
            return Extent2D<F32>{520.0f, 0.0f};
        }
        if (config.content == ModalContent::Dependencies) {
            return Extent2D<F32>{760.0f, 0.0f};
        }
        return Sakura::ModalLayer::Config{}.size;
    }

    Config config;
    std::optional<ModalContent> previousContent;
    AboutView aboutView;
    FlowgraphExamplesView flowgraphExamplesView;
    FlowgraphInfoView flowgraphInfoView;
    SettingsView appSettingsView;
    FlowgraphCloseView flowgraphCloseView;
    RenameBlockView renameBlockView;
    BenchmarkView benchmarkView;
    RemoteView remoteView;
    PluginView pluginView;
    IntelligenceProviderView intelligenceProviderView;
    IntelligenceModelView intelligenceModelView;
    DependencyReviewView dependencyReviewView;
    FeedbackView feedbackView;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH
