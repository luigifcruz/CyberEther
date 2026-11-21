#include "dependencies.hh"

namespace Jetstream {

DependencyReviewModalPresenter::DependencyReviewModalPresenter(const PresenterContext& context) :
    context(context) {}

DependencyReviewView::Config DependencyReviewModalPresenter::build() const {
    DependencyReviewView::Config config;
    const auto& liveRequest = context.state.runtime.dependencyRequest;
    const auto request = context.state.modal.dependencyRequest.value_or(liveRequest);
    const auto& displayRequest = liveRequest.generation == request.generation
                                     ? liveRequest
                                     : request;
    config.stale = context.state.runtime.dependencyGeneration != request.generation;
    config.output = displayRequest.output;
    config.message = displayRequest.message;
    config.approved = displayRequest.approved;

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
        config.dependencies.push_back({dependency.requirement, requestedBy(dependency)});
    }

    const auto enqueue = context.callbacks.enqueueMail;
    config.onInstall = [enqueue, generation = request.generation]() {
        enqueue(MailInstallPythonDependencies{.generation = generation});
    };
    return config;
}

std::string DependencyReviewModalPresenter::flowgraphIdFor(const std::weak_ptr<Flowgraph::View>& weakView) const {
    const auto view = weakView.lock();
    if (!view) {
        return "";
    }
    for (const auto& [id, flowgraph] : context.state.flowgraph.items) {
        if (flowgraph && &flowgraph->view() == view.get()) {
            return id;
        }
    }
    return "";
}

std::string DependencyReviewModalPresenter::requestedBy(const PythonDependencyRequestEntry& dependency) const {
    auto block = dependency.block;
    if (block.ends_with(moduleSuffix)) {
        block.resize(block.size() - moduleSuffix.size());
    }

    auto flowgraphId = flowgraphIdFor(dependency.view);
    if (!flowgraphId.empty() && !block.empty()) {
        return jst::fmt::format("{} / {}", flowgraphId, block);
    }
    return flowgraphId.empty() ? block : flowgraphId;
}

}  // namespace Jetstream
