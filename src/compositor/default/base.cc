#include "base.hh"

#include "actions/base.hh"
#include "presenters/base.hh"
#include "themes.hh"

#include "resources/flowgraphs/base.hh"
#include "resources/fonts/compressed_jbmm.hh"
#include "resources/fonts/compressed_jbmb.hh"
#include "resources/fonts/compressed_fa.hh"

#include "jetstream/platform.hh"
#include "jetstream/registry.hh"
#include "jetstream/logger.hh"

#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

DefaultCompositor::DefaultCompositor()
    : callbacks{
          .enqueueMail = [this](Mail&& mail) {
              enqueue(std::move(mail));
          },
          .enqueueCommand = [this](std::function<Result()> fn, bool silent) {
              Compositor::Impl::enqueue(std::move(fn), silent);
          },
      },
      actions(state, callbacks),
      presenters(state, callbacks) {}

Result DefaultCompositor::create() {
    JST_INFO("[COMPOSITOR_IMPL_DEFAULT] Creating compositor.");

    state.system.instance = instance;
    state.system.render = render;
    state.system.viewport = viewport;

    // Setup theme

    state.sakura.colorMap = themes.at(state.sakura.themeKey);

    // Load example flowgraphs.

    std::vector<Registry::FlowgraphRegistration> manifest;
    const auto res = Resources::GetDefaultManifest(manifest);
    if (res == Result::SUCCESS) {
        for (const auto& entry : manifest) {
            if (Registry::RegisterFlowgraph(entry.key, entry) != Result::SUCCESS) {
                JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to register flowgraph '{}'.", entry.key);
            }
        }
    } else {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Failed to load default flowgraph manifest.");
    }

    // Setup Sakura runtime.

    state.sakura.runtime.update({
        .palette = &state.sakura.colorMap,
        .render = state.system.render.get(),
    });
    state.sakura.runtime.create({
        .body = {.data = jbmm_compressed_data, .size = jbmm_compressed_size},
        .bold = {.data = jbmb_compressed_data, .size = jbmb_compressed_size},
        .iconRegular = {.data = far_compressed_data, .size = far_compressed_size},
        .iconSolid = {.data = fas_compressed_data, .size = fas_compressed_size},
    });

    // Prime the workbench without polling the instance before it is created.

    workbench.update(presenters.build());

    return Result::SUCCESS;
}

Result DefaultCompositor::destroy() {
    JST_INFO("[COMPOSITOR_IMPL_DEFAULT] Destroying compositor.");

    return Result::SUCCESS;
}

Result DefaultCompositor::poll() {
    std::deque<Mail> pending;
    pending.swap(pendingMail);

    // Refresh the action-visible flowgraph list for this poll tick.

    JST_CHECK(state.system.instance->flowgraphList(state.flowgraph.items));

    // Snapshot flowgraph IDs for render-side node context sync.

    flowgraphIds.clear();
    flowgraphIds.reserve(state.flowgraph.items.size());
    for (const auto& [flowgraphId, _] : state.flowgraph.items) {
        flowgraphIds.push_back(flowgraphId);
    }

    // Dispatch the mail tree after snapshots are refreshed.

    for (const auto& mail : pending) {
        if (!state.system.instance) {
            continue;
        }

        JST_CHECK(actions.handle(mail));
    }

    // Surface completed async command results as notifications.

    Command completed;
    while (dequeue(completed)) {
        if (completed.silent) {
            continue;
        }
        Sakura::NotifyResultClean(completed.result, completed.message);
    }

    // Update the state snapshots used by presenters.

    updateWorkbenchState();
    updateFilePendingState();
    updateBenchmarkState();
    updateRemoteState();

    // Build view configs while flowgraph access is confined to poll.

    workbench.update(presenters.build());

    return Result::SUCCESS;
}

Result DefaultCompositor::present() {
    state.sakura.runtime.syncNodeContexts(flowgraphIds);
    state.sakura.runtime.update({
        .palette = &state.sakura.colorMap,
        .render = state.system.render.get(),
    });

    workbench.render(state.sakura.runtime.context());

    return Result::SUCCESS;
}

void DefaultCompositor::updateWorkbenchState() {
    if (state.interface.pendingFocusedFlowgraph.has_value() &&
        state.flowgraph.items.contains(state.interface.pendingFocusedFlowgraph.value())) {
        state.interface.focusedFlowgraph = state.interface.pendingFocusedFlowgraph;
        state.interface.pendingFocusedFlowgraph.reset();
    }

    if (state.flowgraph.items.empty()) {
        state.interface.focusedFlowgraph.reset();
    } else if (state.interface.focusedFlowgraph.has_value() &&
               !state.flowgraph.items.contains(state.interface.focusedFlowgraph.value())) {
        state.interface.focusedFlowgraph.reset();
    }

    if (!state.modal.content.has_value()) {
        state.modal.flowgraph.reset();
        return;
    }

    if (state.modal.content == DefaultCompositorState::ModalState::Content::FlowgraphInfo ||
        state.modal.content == DefaultCompositorState::ModalState::Content::FlowgraphClose) {
        const std::optional<std::string> targetFlowgraph = state.modal.flowgraph.has_value()
            ? state.modal.flowgraph
            : state.interface.focusedFlowgraph;
        if (!targetFlowgraph.has_value() || !state.flowgraph.items.contains(targetFlowgraph.value())) {
            state.modal.content.reset();
            state.modal.flowgraph.reset();
        }
        return;
    }

    if (state.modal.content == DefaultCompositorState::ModalState::Content::RenameBlock &&
        (!state.interface.focusedFlowgraph.has_value() || !state.modal.renameBlockOldName.has_value())) {
        state.modal.content.reset();
        state.modal.renameBlockOldName.reset();
    }
}

void DefaultCompositor::updateFilePendingState() {
#ifdef JST_OS_BROWSER
    state.interface.filePending = Platform::IsFilePending();
#else
    state.interface.filePending = false;
#endif
}

void DefaultCompositor::updateBenchmarkState() {
    if (state.benchmark.running && state.benchmark.future.valid()) {
        const auto status = state.benchmark.future.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
            state.benchmark.future.get();
            state.benchmark.running = false;
        }
    }

    const U64 current = Benchmark::CurrentCount();
    const U64 total = Benchmark::TotalCount();
    state.benchmark.progress = total > 0 ? static_cast<F32>(current) / static_cast<F32>(total) : 0.0f;
    state.benchmark.results = Benchmark::GetResults();
}

void DefaultCompositor::updateRemoteState() {
    const auto remote = state.system.instance->remote();
    const bool remoteStarted = remote->started();
    state.remote.supported = remote->supported();
    state.remote.started = remoteStarted;
    state.remote.clientCount = remoteStarted ? remote->clients().size() : 0;
    state.remote.inviteUrl = remoteStarted ? remote->inviteUrl() : "";
    state.remote.roomId = remoteStarted ? remote->roomId() : "";
    state.remote.accessToken = remoteStarted ? remote->accessToken() : "";
    state.remote.clients = remoteStarted ? remote->clients() : std::vector<Instance::Remote::ClientInfo>{};
    state.remote.waitlist = remoteStarted ? remote->waitlist() : std::vector<std::string>{};
}

void DefaultCompositor::enqueue(Mail&& mail) {
    pendingMail.emplace_back(std::move(mail));
}

std::shared_ptr<Compositor::Impl> DefaultCompositorFactory() {
    return std::make_shared<DefaultCompositor>();
}

}  // namespace Jetstream
