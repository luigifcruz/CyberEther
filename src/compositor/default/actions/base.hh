#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BASE_HH

#include "benchmark.hh"
#include "config_drafts.hh"
#include "crash_report.hh"
#include "dependencies.hh"
#include "feedback.hh"
#include "file_picker.hh"
#include "flowgraph.hh"
#include "gallery.hh"
#include "history.hh"
#include "layout.hh"
#include "live_markdown.hh"
#include "maps.hh"
#include "metrics.hh"
#include "onboarding.hh"
#include "plugins.hh"
#include "remote.hh"
#include "settings.hh"
#include "terminal.hh"
#include "workbench.hh"
#include "agent.hh"

#include "jetstream/logger.hh"

#include <deque>
#include <memory>

namespace Jetstream {

class DefaultActions {
 public:
    DefaultActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks, AgentRuntime& runtime);
    Result dispatch(const std::deque<Mail>& pending);
    Result handle(const Mail& mail);
    Result requestFile(FilePickerRequest request);
    void reconcileFilePicker();
    void settleStartTab();
    void refreshRegistry();
    void settleLinks();
    void settlePendingFocus();
    void reconcileFlowgraphs();
    void reconcileModal();
    void restoreLayouts();
    void reconcileHistory();
    void enqueueFlowgraphEdit(FlowgraphEdit edit);
    void refreshLiveMarkdown();
    void refreshConfigDrafts();
    void refreshMetrics();
    void cancelFilePicker();
    void reconcileAgent();
    void reconcileMaps();
    void shutdownMaps();
    void shutdownAgent();
    void reconcileGallery();
    void shutdownGallery();
    void reconcilePlugins();
    void shutdownPlugins();
    void reconcileTerminals();
    void shutdownTerminals();

 private:
    struct Dispatch;

    std::shared_ptr<BenchmarkActions> benchmark;
    std::shared_ptr<DependencyActions> dependencies;
    std::shared_ptr<FilePickerActions> filePicker;
    std::shared_ptr<FlowgraphActions> flowgraph;
    std::shared_ptr<HistoryActions> history;
    std::shared_ptr<LiveMarkdownActions> liveMarkdown;
    std::shared_ptr<ConfigDraftActions> configDrafts;
    std::shared_ptr<MetricActions> metrics;
    std::shared_ptr<RemoteActions> remote;
    std::shared_ptr<MapsActions> maps;
    std::shared_ptr<SettingsActions> settings;
    std::shared_ptr<LayoutActions> layouts;
    std::shared_ptr<WorkbenchActions> workbench;
    std::shared_ptr<FeedbackActions> feedback;
    std::shared_ptr<OnboardingActions> onboarding;
    std::shared_ptr<CrashReportActions> crashReport;
    std::shared_ptr<AgentActions> agent;
    std::shared_ptr<GalleryActions> gallery;
    std::shared_ptr<PluginActions> plugins;
    std::shared_ptr<TerminalActions> terminals;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BASE_HH
