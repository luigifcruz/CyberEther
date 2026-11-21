#include "history.hh"

namespace Jetstream {

HistoryActions::HistoryActions(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

Result HistoryActions::handle(const MailUndoFlowgraph& msg) {
    return requestSteps(msg.flowgraph, -1);
}

Result HistoryActions::handle(const MailRedoFlowgraph& msg) {
    return requestSteps(msg.flowgraph, 1);
}

Result HistoryActions::handle(const MailRecordFlowgraphLayout& msg) {
    enqueueEdit({.flowgraph = msg.flowgraph, .command = Unchanged, .silent = true, .layout = true});
    return Result::SUCCESS;
}

void HistoryActions::enqueueEdit(FlowgraphEdit edit) {
    const auto entry = state.flowgraph.history.find(edit.flowgraph);
    if (entry != state.flowgraph.history.end() && entry->second.restoring) {
        entry->second.deferred.push_back(std::move(edit));
        return;
    }
    enqueue(std::move(edit));
}

void HistoryActions::reconcile() {
    std::erase_if(state.flowgraph.history, [this](const auto& entry) {
        return !state.flowgraph.items.contains(entry.first);
    });

    for (const auto& [id, flowgraph] : state.flowgraph.items) {
        if (!flowgraph) {
            continue;
        }
        const auto [entry, created] = state.flowgraph.history.try_emplace(id);
        const U64 edits = flowgraph->moduleConfigEdits();
        if (created || entry->second.moduleConfigEdits != edits) {
            entry->second.moduleConfigEdits = edits;
            enqueueEdit({.flowgraph = id, .command = Unchanged, .silent = true});
        }
    }
}

Result HistoryActions::Unchanged() {
    return Result::SUCCESS;
}

void HistoryActions::enqueue(FlowgraphEdit edit) {
    const auto item = state.flowgraph.items.find(edit.flowgraph);
    if (item == state.flowgraph.items.end() || !item->second) {
        callbacks.enqueueCommandWithCompletion(std::move(edit.command), edit.silent, std::move(edit.onComplete));
        return;
    }

    if (const auto entry = state.flowgraph.history.find(edit.flowgraph); entry != state.flowgraph.history.end()) {
        ++entry->second.pendingEdits;
    }
    auto capture = std::make_shared<Capture>();
    callbacks.enqueueCommandWithCompletion([flowgraph = item->second, command = edit.command, capture]() {
        capture->before = CaptureFlowgraphSnapshot(*flowgraph);
        const Result result = command();
        capture->after = CaptureFlowgraphSnapshot(*flowgraph);
        return result;
    }, edit.silent, [this, edit, capture](Result result) {
        if (const auto entry = state.flowgraph.history.find(edit.flowgraph);
            entry != state.flowgraph.history.end() && entry->second.pendingEdits > 0) {
            --entry->second.pendingEdits;
        }
        record(edit, *capture);
        if (edit.onComplete) {
            edit.onComplete(result);
        }
    });
}

void HistoryActions::record(const FlowgraphEdit& edit, const Capture& capture) {
    const auto entry = state.flowgraph.history.find(edit.flowgraph);
    if (entry == state.flowgraph.history.end() || !capture.before || !capture.after) {
        return;
    }
    auto& history = entry->second;

    if (history.entries.empty()) {
        history.seed(capture.before);
    }
    if (ClassifyFlowgraphChange(history.current(), *capture.before) == FlowgraphChange::Content) {
        history.commit(capture.before);
        history.mergeKey.clear();
    } else if (!edit.layout) {
        history.absorb(capture.before);
    }

    if (ClassifyFlowgraphChange(history.current(), *capture.after) == FlowgraphChange::None) {
        history.absorb(capture.after);
        return;
    }

    const bool merges = !edit.mergeKey.empty() && edit.mergeKey == history.mergeKey &&
                        edit.requestedAt - history.mergedAt <= MergeWindow;
    if (merges) {
        history.absorb(capture.after);
    } else {
        history.commit(capture.after);
    }
    history.mergeKey = edit.mergeKey;
    history.mergedAt = edit.requestedAt;
}

Result HistoryActions::requestSteps(const std::string& id, I64 steps) {
    const auto entry = state.flowgraph.history.find(id);
    if (entry == state.flowgraph.history.end()) {
        return Result::SUCCESS;
    }
    auto& history = entry->second;
    if (!history.restoring) {
        history.pendingSteps = steps;
        history.restoring = true;
        settleAfterQueuedEdits(id);
        return Result::SUCCESS;
    }

    auto& deferred = history.deferred;
    if (deferred.empty() && history.pendingSteps != 0) {
        history.pendingSteps += steps;
    } else if (!deferred.empty() && std::holds_alternative<I64>(deferred.back())) {
        std::get<I64>(deferred.back()) += steps;
    } else {
        deferred.emplace_back(steps);
    }
    return Result::SUCCESS;
}

void HistoryActions::settleAfterQueuedEdits(const std::string& id) {
    const auto item = state.flowgraph.items.find(id);
    if (item == state.flowgraph.items.end() || !item->second) {
        callbacks.enqueueCommandWithCompletion(Unchanged, true, [this, id](Result) {
            settle(id, nullptr);
        });
        return;
    }

    auto live = std::make_shared<FlowgraphHistory::Entry>();
    callbacks.enqueueCommandWithCompletion([flowgraph = item->second, live]() {
        *live = CaptureFlowgraphSnapshot(*flowgraph);
        return Result::SUCCESS;
    }, true, [this, id, live](Result) {
        settle(id, *live);
    });
}

void HistoryActions::settle(const std::string& id, const FlowgraphHistory::Entry& live) {
    const auto entry = state.flowgraph.history.find(id);
    const auto item = state.flowgraph.items.find(id);
    if (entry == state.flowgraph.history.end() || item == state.flowgraph.items.end() || !item->second) {
        return;
    }
    auto& history = entry->second;
    if (live) {
        if (history.entries.empty()) {
            history.seed(live);
        } else if (ClassifyFlowgraphChange(history.current(), *live) == FlowgraphChange::Content) {
            history.commit(live);
            history.mergeKey.clear();
        }
    }
    if (history.entries.empty() || !history.settlePendingSteps()) {
        finish(id, history);
        return;
    }
    history.mergeKey.clear();

    auto report = std::make_shared<FlowgraphRestoreReport>();
    callbacks.enqueueCommandWithCompletion([flowgraph = item->second, target = history.entries[history.cursor], report]() {
        const Result result = ApplyFlowgraphSnapshot(*flowgraph, *target, *report);
        report->reached = CaptureFlowgraphSnapshot(*flowgraph);
        return result;
    }, true, [this, id, report](Result result) {
        finishRestore(id, *report, result);
    });
}

void HistoryActions::finishRestore(const std::string& id, const FlowgraphRestoreReport& report, Result result) {
    for (const auto& [oldId, newId] : report.renamed) {
        callbacks.enqueueMail(MailRenameLayoutBlock{.flowgraph = id, .oldId = oldId, .newId = newId});
    }
    for (const auto& block : report.removed) {
        callbacks.enqueueMail(MailRemoveLayoutBlock{.flowgraph = id, .block = block});
    }

    if (result != Result::SUCCESS && result != Result::INCOMPLETE && result != Result::RELOAD) {
        callbacks.notifyResult(result, "");
    }

    const auto entry = state.flowgraph.history.find(id);
    if (entry == state.flowgraph.history.end()) {
        return;
    }
    auto& history = entry->second;
    if (report.reached) {
        auto saved = history.saved;
        const bool drifted = ClassifyFlowgraphChange(history.current(), *report.reached) == FlowgraphChange::Content;
        history.absorb(report.reached);
        if (drifted) {
            history.saved = std::move(saved);
        }
    }
    finish(id, history);
}

void HistoryActions::finish(const std::string& id, FlowgraphHistory& history) {
    history.restoring = false;
    while (!history.deferred.empty()) {
        auto next = std::move(history.deferred.front());
        history.deferred.erase(history.deferred.begin());
        if (auto* edit = std::get_if<FlowgraphEdit>(&next)) {
            enqueue(std::move(*edit));
            continue;
        }
        history.pendingSteps = std::get<I64>(next);
        history.restoring = true;
        settleAfterQueuedEdits(id);
        return;
    }
}

}  // namespace Jetstream
