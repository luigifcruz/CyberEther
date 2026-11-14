#include "base.hh"

namespace Jetstream {

DefaultActions::DefaultActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks, AgentRuntime& runtime) {
    workbench = std::make_shared<WorkbenchActions>(state, callbacks);
    settings = std::make_shared<SettingsActions>(state, callbacks);
    benchmark = std::make_shared<BenchmarkActions>(state, callbacks);
    remote = std::make_shared<RemoteActions>(state, callbacks);
    maps = std::make_shared<MapsActions>(state, callbacks);
    filePicker = std::make_shared<FilePickerActions>(state, callbacks);
    flowgraph = std::make_shared<FlowgraphActions>(state, callbacks);
    dependencies = std::make_shared<DependencyActions>(callbacks);
    history = std::make_shared<HistoryActions>(state, callbacks);
    layouts = std::make_shared<LayoutActions>(state.flowgraph);
    liveMarkdown = std::make_shared<LiveMarkdownActions>(state.flowgraph);
    configDrafts = std::make_shared<ConfigDraftActions>(state.flowgraph);
    metrics = std::make_shared<MetricActions>(state.flowgraph);
    feedback = std::make_shared<FeedbackActions>(state, callbacks);
    onboarding = std::make_shared<OnboardingActions>(state);
    agent = std::make_shared<AgentActions>(state, callbacks, runtime);
    gallery = std::make_shared<GalleryActions>(state);
    terminals = std::make_shared<TerminalActions>(state, callbacks);
}

Result DefaultActions::dispatch(const std::deque<Mail>& pending) {
    for (U64 index = 0; index < pending.size(); ++index) {
        const auto& mail = pending[index];
        if (std::holds_alternative<MailSaveFlowgraph>(mail)) {
            // Apply trailing presentation reports before queuing the save,
            // without moving them across intervening user actions.
            while (index + 1 < pending.size()) {
                const auto& report = pending[index + 1];
                if (!std::holds_alternative<MailSetFlowgraphLayout>(report) &&
                    !std::holds_alternative<MailSetNodeMeta>(report) &&
                    !std::holds_alternative<MailSetEditorMeta>(report) &&
                    !std::holds_alternative<MailResizeSurface>(report)) {
                    break;
                }
                JST_CHECK(handle(report));
                ++index;
            }
        }
        JST_CHECK(handle(mail));
    }
    return Result::SUCCESS;
}

Result DefaultActions::handle(const Mail& mail) {
    Result result = Result::SUCCESS;
    if (Handle(*workbench, mail, result)) {
        return result;
    }
    if (Handle(*settings, mail, result)) {
        return result;
    }
    if (Handle(*benchmark, mail, result)) {
        return result;
    }
    if (Handle(*remote, mail, result)) {
        return result;
    }
    if (Handle(*maps, mail, result)) {
        return result;
    }
    if (Handle(*filePicker, mail, result)) {
        return result;
    }
    if (Handle(*flowgraph, mail, result)) {
        return result;
    }
    if (Handle(*dependencies, mail, result)) {
        return result;
    }
    if (Handle(*history, mail, result)) {
        return result;
    }
    if (Handle(*layouts, mail, result)) {
        return result;
    }
    if (Handle(*configDrafts, mail, result)) {
        return result;
    }
    if (Handle(*feedback, mail, result)) {
        return result;
    }
    if (Handle(*onboarding, mail, result)) {
        return result;
    }
    if (Handle(*agent, mail, result)) {
        return result;
    }
    if (Handle(*terminals, mail, result)) {
        return result;
    }

    JST_ERROR("[COMPOSITOR_IMPL_DEFAULT] Unhandled mail.");
    return Result::ERROR;
}

Result DefaultActions::requestFile(FilePickerRequest request) {
    return filePicker->request(std::move(request));
}

void DefaultActions::reconcileFilePicker() {
    filePicker->reconcileAvailability();
}

void DefaultActions::settleStartTab() {
    workbench->settleStartTab();
}

void DefaultActions::refreshRegistry() {
    gallery->refreshRegistry();
}

void DefaultActions::settlePendingFocus() {
    workbench->settlePendingFocus();
}

void DefaultActions::reconcileFlowgraphs() {
    workbench->reconcileFlowgraphs();
}

void DefaultActions::reconcileModal() {
    workbench->reconcileModal();
}

void DefaultActions::restoreLayouts() {
    layouts->restoreFromMetadata();
}

void DefaultActions::reconcileHistory() {
    history->reconcile();
}

void DefaultActions::enqueueFlowgraphEdit(FlowgraphEdit edit) {
    history->enqueueEdit(std::move(edit));
}

void DefaultActions::refreshLiveMarkdown() {
    liveMarkdown->refresh();
}

void DefaultActions::refreshConfigDrafts() {
    configDrafts->refresh();
}

void DefaultActions::refreshMetrics() {
    metrics->refresh();
}

void DefaultActions::cancelFilePicker() {
    filePicker->cancel();
}

void DefaultActions::reconcileAgent() {
    agent->reconcile();
}

void DefaultActions::reconcileMaps() {
    maps->reconcile();
}

void DefaultActions::shutdownMaps() {
    maps->shutdown();
}

void DefaultActions::shutdownAgent() {
    agent->shutdown();
}

void DefaultActions::reconcileGallery() {
    gallery->reconcile();
}

void DefaultActions::shutdownGallery() {
    gallery->shutdown();
}

void DefaultActions::reconcileTerminals() {
    terminals->reconcile();
}

void DefaultActions::shutdownTerminals() {
    terminals->shutdown();
}

}  // namespace Jetstream
