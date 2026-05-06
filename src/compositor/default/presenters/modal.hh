#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_HH

#include "benchmark.hh"
#include "flowgraph.hh"
#include "remote.hh"
#include "settings.hh"

#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"
#include "../views/modal/container.hh"

#include <memory>
#include <utility>

namespace Jetstream {

struct DefaultModalPresenter {
    using ModalContent = DefaultCompositorState::ModalState::Content;

    DefaultModalPresenter(DefaultCompositorState& state,
                          DefaultCompositorCallbacks& callbacks) : state(state),
                                                                   callbacks(callbacks) {
        benchmark = std::make_shared<DefaultBenchmarkPresenter>(this->state, this->callbacks);
        remote = std::make_shared<DefaultRemotePresenter>(this->state, this->callbacks);
        settings = std::make_shared<DefaultSettingsPresenter>(this->state, this->callbacks);
    }

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    ModalView::Config build(DefaultFlowgraphPresenter::ModalConfigs flowgraphModals) const {
        const auto enqueue = callbacks.enqueueMail;
        ModalView::Config modalConfig;
        modalConfig.content = state.modal.content;
        modalConfig.onClose = [enqueue]() {
            enqueue(MailCloseModal{});
        };

        if (state.modal.content == ModalContent::FlowgraphExamples) {
            modalConfig.flowgraphExamples = std::move(flowgraphModals.examples);
        } else if (state.modal.content == ModalContent::FlowgraphInfo) {
            modalConfig.flowgraphInfo = std::move(flowgraphModals.info);
        } else if (state.modal.content == ModalContent::Settings) {
            modalConfig.appSettings = settings->build();
        } else if (state.modal.content == ModalContent::FlowgraphClose) {
            modalConfig.flowgraphClose = std::move(flowgraphModals.close);
        } else if (state.modal.content == ModalContent::RenameBlock) {
            modalConfig.renameBlock = std::move(flowgraphModals.renameBlock);
        } else if (state.modal.content == ModalContent::Benchmark) {
            modalConfig.benchmark = benchmark->build();
        } else if (state.modal.content == ModalContent::RemoteStreaming) {
            modalConfig.remoteStreaming = remote->build();
        }

        return modalConfig;
    }

 private:
    std::shared_ptr<DefaultBenchmarkPresenter> benchmark;
    std::shared_ptr<DefaultRemotePresenter> remote;
    std::shared_ptr<DefaultSettingsPresenter> settings;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_HH
