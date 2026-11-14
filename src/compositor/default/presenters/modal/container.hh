#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CONTAINER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CONTAINER_HH

#include "benchmark.hh"
#include "dependencies.hh"
#include "docs_search.hh"
#include "feedback.hh"
#include "onboarding.hh"
#include "remote_qr.hh"

#include "../context.hh"
#include "../flowgraph/modals/info.hh"
#include "../flowgraph/modals/rename.hh"

#include "../../model/messages.hh"
#include "../../views/modal/container.hh"

namespace Jetstream {

struct ModalPresenter {
    const PresenterContext& context;
    FlowgraphInfoModalPresenter flowgraphInfo;
    RenameBlockModalPresenter renameBlock;
    BenchmarkModalPresenter benchmark;
    DependencyReviewModalPresenter dependencies;
    FeedbackModalPresenter feedback;
    OnboardingModalPresenter onboarding;
    RemoteQrModalPresenter remoteQr;
    DocsSearchModalPresenter docsSearch;

    explicit ModalPresenter(const PresenterContext& context);
    ModalView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CONTAINER_HH
