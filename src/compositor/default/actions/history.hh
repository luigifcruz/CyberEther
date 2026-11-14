#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_HISTORY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_HISTORY_HH

#include "../support/history.hh"
#include "../model/callbacks.hh"
#include "../model/history.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include <chrono>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace Jetstream {

struct HistoryActions {
    using Filter = std::tuple<MailUndoFlowgraph,
                              MailRedoFlowgraph,
                              MailRecordFlowgraphLayout>;
    using Clock = FlowgraphHistory::Clock;

    static constexpr std::chrono::milliseconds MergeWindow{1000};

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    HistoryActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks);
    Result handle(const MailUndoFlowgraph& msg);
    Result handle(const MailRedoFlowgraph& msg);
    Result handle(const MailRecordFlowgraphLayout& msg);
    void enqueueEdit(FlowgraphEdit edit);
    void reconcile();

 private:
    struct Capture {
        FlowgraphHistory::Entry before;
        FlowgraphHistory::Entry after;
    };

    static Result Unchanged();
    void enqueue(FlowgraphEdit edit);
    void record(const FlowgraphEdit& edit, const Capture& capture);
    Result requestSteps(const std::string& id, I64 steps);
    void settleAfterQueuedEdits(const std::string& id);
    void settle(const std::string& id, const FlowgraphHistory::Entry& live);
    void finishRestore(const std::string& id, const FlowgraphRestoreReport& report, Result result);
    void finish(const std::string& id, FlowgraphHistory& history);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_HISTORY_HH
