#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH

#include "about.hh"
#include "benchmark.hh"
#include "dependencies.hh"
#include "docs_search.hh"
#include "../flowgraph/modals/info.hh"
#include "../flowgraph/modals/rename.hh"
#include "feedback.hh"
#include "onboarding.hh"
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
        OnboardingView::Config onboarding;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    Sakura::ModalLayer::Config layer();

 private:
    Sakura::Component* content();
    Extent2D<F32> modalSize() const;

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
    OnboardingView onboardingView;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CONTAINER_HH
