#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BASE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BASE_HH

#include "benchmark.hh"
#include "dependencies.hh"
#include "feedback.hh"
#include "file_picker.hh"
#include "flowgraph.hh"
#include "layout.hh"
#include "live_markdown.hh"
#include "metrics.hh"
#include "remote.hh"
#include "settings.hh"
#include "workbench.hh"

#include "jetstream/logger.hh"

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace Jetstream {

class DefaultActions {
 public:
    DefaultActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks) {
        workbench = std::make_shared<WorkbenchActions>(state, callbacks);
        settings = std::make_shared<SettingsActions>(state, callbacks);
        benchmark = std::make_shared<BenchmarkActions>(state, callbacks);
        remote = std::make_shared<RemoteActions>(state, callbacks);
        filePicker = std::make_shared<FilePickerActions>(state, callbacks);
        flowgraph = std::make_shared<FlowgraphActions>(state, callbacks);
        dependencies = std::make_shared<DependencyActions>(callbacks);
        layouts = std::make_shared<LayoutActions>(state.flowgraph);
        liveMarkdown = std::make_shared<LiveMarkdownActions>(state.flowgraph);
        metrics = std::make_shared<MetricActions>(state.flowgraph);
        feedback = std::make_shared<FeedbackActions>(state, callbacks);
    }

    Result handle(const Mail& mail) {
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
        if (Handle(*filePicker, mail, result)) {
            return result;
        }
        if (Handle(*flowgraph, mail, result)) {
            return result;
        }
        if (Handle(*dependencies, mail, result)) {
            return result;
        }
        if (Handle(*layouts, mail, result)) {
            return result;
        }
        if (Handle(*feedback, mail, result)) {
            return result;
        }

        JST_ERROR("[COMPOSITOR_IMPL_DEFAULT] Unhandled mail.");
        return Result::ERROR;
    }

    Result requestFile(FilePickerRequest request) {
        return filePicker->request(std::move(request));
    }

    void reconcileFilePicker() {
        filePicker->reconcileAvailability();
    }

    void settleStartTab() {
        workbench->settleStartTab();
    }

    void restoreLayouts() {
        layouts->restoreFromMetadata();
    }

    void refreshLiveMarkdown() {
        liveMarkdown->refresh();
    }

    void refreshMetrics() {
        metrics->refresh();
    }

    void cancelFilePicker() {
        filePicker->cancel();
    }

 private:
    template<class Action, class... Signals>
    static bool Handle(Action& action, const Mail& mail, Result& result, std::tuple<Signals...>*) {
        return std::visit([&](const auto& msg) {
            using Signal = std::decay_t<decltype(msg)>;
            if constexpr ((std::is_same_v<Signal, Signals> || ...)) {
                result = action.handle(msg);
                return true;
            }
            return false;
        }, mail);
    }

    template<class Action>
    static bool Handle(Action& action, const Mail& mail, Result& result) {
        return Handle(action, mail, result, static_cast<typename Action::Filter*>(nullptr));
    }

    std::shared_ptr<BenchmarkActions> benchmark;
    std::shared_ptr<DependencyActions> dependencies;
    std::shared_ptr<FilePickerActions> filePicker;
    std::shared_ptr<FlowgraphActions> flowgraph;
    std::shared_ptr<LiveMarkdownActions> liveMarkdown;
    std::shared_ptr<MetricActions> metrics;
    std::shared_ptr<RemoteActions> remote;
    std::shared_ptr<SettingsActions> settings;
    std::shared_ptr<LayoutActions> layouts;
    std::shared_ptr<WorkbenchActions> workbench;
    std::shared_ptr<FeedbackActions> feedback;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_BASE_HH
