#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH

#include "about.hh"
#include "benchmark.hh"
#include "../flowgraph/modals/close.hh"
#include "../flowgraph/modals/examples.hh"
#include "../flowgraph/modals/info.hh"
#include "../flowgraph/modals/rename.hh"
#include "settings/base.hh"
#include "../../model/ui.hh"
#include "remote.hh"

#include "jetstream/render/sakura/sakura.hh"

#include <functional>
#include <optional>
#include <utility>

namespace Jetstream {

struct ModalView : public Sakura::Component {
    struct Config {
        std::optional<ModalContent> content;
        FlowgraphExamplesView::Config flowgraphExamples;
        std::optional<FlowgraphInfoView::Config> flowgraphInfo;
        SettingsView::Config appSettings;
        std::optional<FlowgraphCloseView::Config> flowgraphClose;
        std::optional<RenameBlockView::Config> renameBlock;
        BenchmarkView::Config benchmark;
        RemoteView::Config remoteStreaming;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);
        modal.update({
            .id = "GlobalModal",
            .size = modalSize(),
            .onClose = [this]() {
                if (this->config.onClose) {
                    this->config.onClose();
                }
            },
        });

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
        }

        previousContent = this->config.content;
    }

    void render(const Sakura::Context& ctx) {
        if (!config.content.has_value()) {
            return;
        }

        modal.render(ctx, [this](const Sakura::Context& ctx) {
            switch (config.content.value()) {
                case ModalContent::About:
                    aboutView.render(ctx);
                    break;
                case ModalContent::FlowgraphExamples:
                    flowgraphExamplesView.render(ctx);
                    break;
                case ModalContent::FlowgraphInfo:
                    if (config.flowgraphInfo.has_value()) {
                        flowgraphInfoView.render(ctx);
                    }
                    break;
                case ModalContent::Settings:
                    appSettingsView.render(ctx);
                    break;
                case ModalContent::FlowgraphClose:
                    if (config.flowgraphClose.has_value()) {
                        flowgraphCloseView.render(ctx);
                    }
                    break;
                case ModalContent::RenameBlock:
                    if (config.renameBlock.has_value()) {
                        renameBlockView.render(ctx);
                    }
                    break;
                case ModalContent::Benchmark:
                    benchmarkView.render(ctx);
                    break;
                case ModalContent::RemoteStreaming:
                    remoteView.render(ctx);
                    break;
            }
        });
    }

 private:
    std::optional<Extent2D<F32>> modalSize() const {
        if (config.content == ModalContent::Settings) {
            return Extent2D<F32>{880.0f, 700.0f};
        }
        return std::nullopt;
    }

    Config config;
    std::optional<ModalContent> previousContent;
    Sakura::Modal modal;
    AboutView aboutView;
    FlowgraphExamplesView flowgraphExamplesView;
    FlowgraphInfoView flowgraphInfoView;
    SettingsView appSettingsView;
    FlowgraphCloseView flowgraphCloseView;
    RenameBlockView renameBlockView;
    BenchmarkView benchmarkView;
    RemoteView remoteView;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH
