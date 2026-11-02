#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CONTAINER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CONTAINER_HH

#include "benchmark.hh"
#include "dependencies.hh"
#include "docs_search.hh"
#include "feedback.hh"
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
    RemoteQrModalPresenter remoteQr;
    DocsSearchModalPresenter docsSearch;

    explicit ModalPresenter(const PresenterContext& context) : context(context),
                                                               flowgraphInfo(context),
                                                               renameBlock(context),
                                                               benchmark(context),
                                                               dependencies(context),
                                                               feedback(context),
                                                               remoteQr(context),
                                                               docsSearch(context) {}

    ModalView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;

        ModalView::Config config;
        config.content = context.state.modal.content;
        config.onClose = [enqueue]() {
            enqueue(MailCloseModal{});
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
        }

        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CONTAINER_HH
