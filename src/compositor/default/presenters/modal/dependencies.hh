#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DEPENDENCIES_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DEPENDENCIES_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/dependencies.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct DependencyReviewModalPresenter {
    const PresenterContext& context;

    explicit DependencyReviewModalPresenter(const PresenterContext& context) :
        context(context) {}

    DependencyReviewView::Config build() const {
        DependencyReviewView::Config config;
        const auto& liveRequest = context.state.runtime.dependencyRequest;
        const auto request = context.state.modal.dependencyRequest.value_or(liveRequest);
        const auto& displayRequest = liveRequest.generation == request.generation
                                         ? liveRequest
                                         : request;
        config.stale = context.state.runtime.dependencyGeneration != request.generation;
        config.output = displayRequest.output;
        config.message = displayRequest.message;

        switch (displayRequest.state) {
            case PythonDependencyRequestState::ApprovalRequired:
                config.state = DependencyReviewView::State::Review;
                break;
            case PythonDependencyRequestState::Denied:
                config.state = DependencyReviewView::State::Denied;
                break;
            case PythonDependencyRequestState::Installing:
                config.state = DependencyReviewView::State::Installing;
                break;
            case PythonDependencyRequestState::Installed:
                config.state = DependencyReviewView::State::Installed;
                break;
            case PythonDependencyRequestState::Failed:
                config.state = DependencyReviewView::State::Failed;
                break;
            case PythonDependencyRequestState::None:
                break;
        }
        config.installEnabled = !config.stale && !request.dependencies.empty() &&
                                (displayRequest.state ==
                                     PythonDependencyRequestState::ApprovalRequired ||
                                 displayRequest.state ==
                                     PythonDependencyRequestState::Failed);

        config.dependencies.reserve(request.dependencies.size());
        for (const auto& dependency : request.dependencies) {
            auto block = dependency.block;
            if (block.ends_with("-python")) {
                block.resize(block.size() - std::string("-python").size());
            }

            std::string flowgraphId;
            if (const auto view = dependency.view.lock()) {
                for (const auto& [id, flowgraph] : context.state.flowgraph.items) {
                    if (flowgraph && &flowgraph->view() == view.get()) {
                        flowgraphId = id;
                        break;
                    }
                }
            }

            std::string requestedBy;
            if (!flowgraphId.empty() && !block.empty()) {
                requestedBy = flowgraphId + " / " + block;
            } else if (!flowgraphId.empty()) {
                requestedBy = std::move(flowgraphId);
            } else {
                requestedBy = std::move(block);
            }
            config.dependencies.push_back({dependency.requirement, std::move(requestedBy)});
        }

        const auto enqueue = context.callbacks.enqueueMail;
        config.onInstall = [enqueue, generation = request.generation]() {
            enqueue(MailInstallPythonDependencies{.generation = generation});
        };
        return config;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_DEPENDENCIES_HH
