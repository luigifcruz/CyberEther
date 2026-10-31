#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH

#include "about.hh"
#include "benchmark.hh"
#include "dependencies.hh"
#include "docs_search.hh"
#include "../flowgraph/modals/info.hh"
#include "../flowgraph/modals/rename.hh"
#include "feedback.hh"
#include "remote_qr.hh"
#include "../../model/ui.hh"

#include "jetstream/render/sakura/base.hh"

#include <optional>
#include <utility>

namespace Jetstream {

struct ModalView {
    struct Config {
        std::optional<ModalContent> content;
        std::optional<FlowgraphInfoView::Config> flowgraphInfo;
        std::optional<RenameBlockView::Config> renameBlock;
        BenchmarkView::Config benchmark;
        DependencyReviewView::Config dependencies;
        FeedbackView::Config feedback;
        RemoteQrView::Config remoteQr;
        DocsSearchView::Config docsSearch;
        Sakura::Callback<void()> onClose;
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
        }

        previousContent = this->config.content;
    }

    Sakura::ModalLayer::Config layer() {
        if (config.content == ModalContent::DocsSearch) {
            return docsSearchView.layer();
        }
        return {
            .content = content(),
            .size = modalSize(),
            .closeButton = config.content != ModalContent::Feedback,
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
        }
        return nullptr;
    }

    Extent2D<F32> modalSize() const {
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
    FlowgraphInfoView flowgraphInfoView;
    RenameBlockView renameBlockView;
    BenchmarkView benchmarkView;
    DependencyReviewView dependencyReviewView;
    FeedbackView feedbackView;
    RemoteQrView remoteQrView;
    DocsSearchView docsSearchView;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH
