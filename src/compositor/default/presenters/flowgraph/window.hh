#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_WINDOW_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_WINDOW_HH

#include "editor/base.hh"
#include "labels.hh"
#include "stack.hh"
#include "surface.hh"

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/components/callout.hh"
#include "../../views/flowgraph/window.hh"

#include "jetstream/flowgraph.hh"
#include "jetstream/flowgraph_view.hh"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphWindowPresenter {
    const PresenterContext& context;
    FlowgraphEditorPresenter editor;
    StackPresenter stacks;
    FlowgraphDetachedSurfacePresenter surfaces;

    explicit FlowgraphWindowPresenter(const PresenterContext& context) : context(context),
                                                                          editor(context),
                                                                          stacks(context),
                                                                          surfaces(context) {}

    FlowgraphWindow::Config build(const std::string& flowgraphId,
                                  const std::shared_ptr<Flowgraph>& flowgraph) const {
        const auto enqueue = context.callbacks.enqueueMail;
        std::string title = MakeFlowgraphWindowTitle(flowgraphId, flowgraph);
        if (context.state.interface.focusedFlowgraph == flowgraphId) {
            title = "• " + title;
        }
        return FlowgraphWindow::Config{
            .id = MakeFlowgraphWindowId(flowgraphId),
            .title = std::move(title),
            .editor = editor.build(flowgraphId, flowgraph),
            .stacks = stacks.build(flowgraphId, flowgraph),
            .detachedSurfaces = surfaces.build(flowgraphId, flowgraph),
            .empty = flowgraph->view().empty(),
            .dependencyReviewAvailable = hasDependencyReview(flowgraph),
            .dependencyReviewTone = dependencyReviewTone(),
            .dependencyReviewMessage = dependencyReviewMessage(),
            .onFocus = [enqueue, flowgraphId]() {
                enqueue(MailFocusFlowgraph{flowgraphId});
            },
            .onSave = [enqueue, flowgraphId]() {
                enqueue(MailSaveFlowgraph{.flowgraph = flowgraphId});
            },
            .onClose = [enqueue, flowgraphId]() {
                enqueue(MailCloseFlowgraph{flowgraphId});
            },
            .onCreateStack = [enqueue, flowgraphId]() {
                enqueue(MailCreateStack{.flowgraph = flowgraphId});
            },
            .onSendFeedback = [enqueue]() {
                enqueue(MailOpenModal{.content = ModalContent::Feedback});
            },
            .onReviewDependencies = [enqueue, flowgraphId]() {
                enqueue(MailOpenModal{
                    .content = ModalContent::Dependencies,
                    .flowgraph = flowgraphId,
                });
            },
        };
    }

 private:
    Callout::Tone dependencyReviewTone() const {
        switch (context.state.runtime.dependencyRequest.state) {
            case PythonDependencyRequestState::Failed:
                return Callout::Tone::Error;
            case PythonDependencyRequestState::Denied:
                return Callout::Tone::Warning;
            default:
                return Callout::Tone::Info;
        }
    }

    std::string dependencyReviewMessage() const {
        switch (context.state.runtime.dependencyRequest.state) {
            case PythonDependencyRequestState::Failed:
                return "Python dependency installation failed.";
            case PythonDependencyRequestState::Denied:
                return "Python dependencies are blocked by policy.";
            default:
                return "Python dependencies need approval.";
        }
    }

    bool hasDependencyReview(const std::shared_ptr<Flowgraph>& currentFlowgraph) const {
        const auto& request = context.state.runtime.dependencyRequest;
        if (request.state != PythonDependencyRequestState::ApprovalRequired &&
            request.state != PythonDependencyRequestState::Failed &&
            request.state != PythonDependencyRequestState::Denied) {
            return false;
        }

        const auto* currentView = &currentFlowgraph->view();
        return !request.dependencies.empty() &&
               std::ranges::any_of(
                   request.dependencies,
                   [currentView](const auto& dependency) {
                       const auto view = dependency.view.lock();
                       return !view || view.get() == currentView;
                   });
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_WINDOW_HH
