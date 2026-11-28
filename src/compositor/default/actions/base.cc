#include "base.hh"

#include <array>
#include <utility>

namespace Jetstream {

struct DefaultActions::Dispatch {
    using Route = Result (*)(DefaultActions& actions, const Mail& mail);

    template<typename Member>
    struct Owner;

    template<typename Action>
    struct Owner<std::shared_ptr<Action> DefaultActions::*> {
        using Type = Action;
    };

    template<typename T, auto Member>
    static constexpr bool Handles = requires(typename Owner<decltype(Member)>::Type& action, const T& mail) {
        action.handle(mail);
    };

    template<auto... Members>
    struct Over {
        template<typename T>
        static consteval U64 Handlers() {
            constexpr bool handles[] = {Handles<T, Members>...};
            U64 count = 0;
            for (const bool handled : handles) {
                count += handled ? 1 : 0;
            }
            return count;
        }

        template<typename T, auto Member>
        static bool Invoke(DefaultActions& actions, const T& mail, Result& result) {
            if constexpr (Handles<T, Member>) {
                result = (*(actions.*Member)).handle(mail);
                return true;
            } else {
                return false;
            }
        }

        template<typename T>
        static Result Deliver(DefaultActions& actions, const Mail& mail) {
            static_assert(Handlers<T>() == 1, "Every Mail type needs exactly one action handler");
            const T& payload = *mail.getIf<T>();
            Result result = Result::ERROR;
            (Invoke<T, Members>(actions, payload, result) || ...);
            return result;
        }

        template<typename... Ts>
        struct Table {
            static constexpr std::array<Route, sizeof...(Ts)> Routes = {&Deliver<Ts>...};
        };
    };

    using Actions = Over<&DefaultActions::workbench,
                         &DefaultActions::settings,
                         &DefaultActions::plugins,
                         &DefaultActions::benchmark,
                         &DefaultActions::remote,
                         &DefaultActions::maps,
                         &DefaultActions::filePicker,
                         &DefaultActions::flowgraph,
                         &DefaultActions::dependencies,
                         &DefaultActions::history,
                         &DefaultActions::layouts,
                         &DefaultActions::configDrafts,
                         &DefaultActions::feedback,
                         &DefaultActions::onboarding,
                         &DefaultActions::crashReport,
                         &DefaultActions::agent,
                         &DefaultActions::terminals>;

    static constexpr const auto& Routes = Mail::Expand<Actions::Table>::Routes;

    static_assert(Routes.size() == Mail::Count);
};

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
    crashReport = std::make_shared<CrashReportActions>(state, callbacks);
    agent = std::make_shared<AgentActions>(state, callbacks, runtime);
    gallery = std::make_shared<GalleryActions>(state, Home::Get().gallery());
    plugins = std::make_shared<PluginActions>(state, callbacks);
    terminals = std::make_shared<TerminalActions>(state, callbacks);
}

Result DefaultActions::dispatch(const std::deque<Mail>& pending) {
    for (U64 index = 0; index < pending.size(); ++index) {
        const auto& mail = pending[index];
        if (mail.is<MailSaveFlowgraph>()) {
            // Apply trailing presentation reports before queuing the save,
            // without moving them across intervening user actions.
            while (index + 1 < pending.size()) {
                const auto& report = pending[index + 1];
                if (!report.isAnyOf<MailSetFlowgraphLayout, MailSetNodeMeta, MailSetEditorMeta, MailResizeSurface>()) {
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
    if (mail.empty()) {
        JST_ERROR("[COMPOSITOR_IMPL_DEFAULT] Cannot dispatch empty mail.");
        return Result::ERROR;
    }
    return Dispatch::Routes[mail.index()](*this, mail);
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

void DefaultActions::settleLinks() {
    workbench->settleLinks();
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

void DefaultActions::reconcilePlugins() {
    plugins->reconcile();
}

void DefaultActions::shutdownPlugins() {
    plugins->shutdown();
}

void DefaultActions::reconcileTerminals() {
    terminals->reconcile();
}

void DefaultActions::shutdownTerminals() {
    terminals->shutdown();
}

}  // namespace Jetstream
