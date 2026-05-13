#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CONTAINER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CONTAINER_HH

#include "benchmark.hh"
#include "library.hh"
#include "remote.hh"
#include "settings/base.hh"

#include "../context.hh"
#include "../flowgraph/modals/close.hh"
#include "../flowgraph/modals/examples.hh"
#include "../flowgraph/modals/info.hh"
#include "../flowgraph/modals/rename.hh"

#include "../../model/messages.hh"
#include "../../views/modal/container.hh"

namespace Jetstream {

struct ModalPresenter {
    const PresenterContext& context;
    SettingsModalPresenter settings;
    FlowgraphExamplesModalPresenter flowgraphExamples;
    FlowgraphInfoModalPresenter flowgraphInfo;
    FlowgraphCloseModalPresenter flowgraphClose;
    RenameBlockModalPresenter renameBlock;
    BenchmarkModalPresenter benchmark;
    RemoteStreamingModalPresenter remoteStreaming;
    LibraryPresenter library;

    explicit ModalPresenter(const PresenterContext& context) : context(context),
                                                               settings(context),
                                                               flowgraphExamples(context),
                                                               flowgraphInfo(context),
                                                               flowgraphClose(context),
                                                               renameBlock(context),
                                                               benchmark(context),
                                                               remoteStreaming(context),
                                                               library(context) {}

    ModalView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;

        ModalView::Config config;
        config.content = context.state.modal.content;
        config.onClose = [enqueue, content = config.content]() {
            if (content == ModalContent::Library) {
                enqueue(MailOpenModal{.content = ModalContent::Settings, .settings = SettingsSection::Registry});
                return;
            }
            enqueue(MailCloseModal{});
        };

        if (!config.content.has_value()) {
            return config;
        }

        switch (config.content.value()) {
            case ModalContent::About:
                break;
            case ModalContent::FlowgraphExamples:
                config.flowgraphExamples = flowgraphExamples.build();
                break;
            case ModalContent::FlowgraphInfo:
                config.flowgraphInfo = flowgraphInfo.build();
                break;
            case ModalContent::FlowgraphClose:
                config.flowgraphClose = flowgraphClose.build();
                break;
            case ModalContent::RenameBlock:
                config.renameBlock = renameBlock.build();
                break;
            case ModalContent::Benchmark:
                config.benchmark = benchmark.build();
                break;
            case ModalContent::RemoteStreaming:
                config.remoteStreaming = remoteStreaming.build();
                break;
            case ModalContent::Settings:
                config.appSettings = settings.build();
                break;
            case ModalContent::Library:
                config.library = library.build();
                break;
        }

        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_CONTAINER_HH
