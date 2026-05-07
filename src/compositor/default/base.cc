#include "base.hh"

#include "actions/base.hh"
#include "presenters/base.hh"
#include "themes.hh"

#include "resources/fonts/compressed_jbmm.hh"
#include "resources/fonts/compressed_jbmb.hh"
#include "resources/fonts/compressed_fa.hh"

#include "jetstream/platform.hh"
#include "jetstream/logger.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/settings.hh"

#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

DefaultCompositor::DefaultCompositor() :
    callbacks{
        .enqueueMail = [this](Mail&& mail) {
            enqueue(std::move(mail));
        },
        .enqueueCommand = [this](std::function<Result()> fn, bool silent) {
            Compositor::Impl::enqueue(std::move(fn), silent);
        },
        .notify = [](Sakura::ToastType type, I32 durationMs, const std::string& message) {
            Sakura::PushToast(type, durationMs, message);
        },
        .notifyResult = [](Result result, const std::string& message) {
            Sakura::PushToastResult(result, message);
        },
        .setClipboardText = [](const std::string& value) {
            Sakura::SetClipboardText(value);
        },
    },
    actions(state, callbacks),
    presenters(state, callbacks) {}

Result DefaultCompositor::create() {
    JST_INFO("[COMPOSITOR_IMPL_DEFAULT] Creating compositor.");

    // Bind compositor-owned runtime services.

    state.system.instance = instance;
    state.system.render = render;
    state.system.viewport = viewport;

    // Load persisted application settings.

    Settings settings;
    JST_CHECK(Settings::Get(settings));

    // Restore Sakura theme state.

    state.sakura.themeKey = themes.contains(settings.interface.themeKey)
        ? settings.interface.themeKey
        : "Dark";
    state.sakura.colorMap = themes.at(state.sakura.themeKey);

    // Restore graphics preferences.

    state.graphics.device = settings.graphics.device;
    state.graphics.scale = settings.graphics.scale;
    state.graphics.framerate = settings.graphics.framerate;

    // Restore interface preferences.

    state.interface.infoPanelEnabled = settings.interface.infoPanelEnabled;
    state.interface.backgroundParticles = settings.interface.backgroundParticles;

    // Restore debug preferences.

    state.debug.logLevel = settings.developer.logLevel;
    state.debug.latencyEnabled = settings.developer.latencyEnabled;
    state.debug.runtimeMetricsEnabled = settings.developer.runtimeMetricsEnabled;
    JST_LOG_SET_DEBUG_LEVEL(state.debug.logLevel);

    // Restore remote streaming preferences.

    state.remote.brokerUrl = settings.remote.brokerUrl;
    state.remote.autoJoinSessions = settings.remote.autoJoinSessions;
    state.remote.framerate = static_cast<U32>(settings.remote.framerate);

    try {
        state.remote.codec = StringToRemoteCodec(settings.remote.codec);
    } catch (const Result&) {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Invalid saved remote codec '{}'. Using default.",
                 settings.remote.codec);
    }

    try {
        state.remote.encoder = StringToRemoteEncoder(settings.remote.encoder);
    } catch (const Result&) {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Invalid saved remote encoder '{}'. Using default.",
                 settings.remote.encoder);
    }

    // Initialize Sakura rendering resources.

    state.sakura.runtime.update({
        .palette = &state.sakura.colorMap,
        .render = state.system.render.get(),
    });
    state.sakura.runtime.create({
        .body = {
            .data = jbmm_compressed_data,
            .size = jbmm_compressed_size,
        },
        .bold = {
            .data = jbmb_compressed_data,
            .size = jbmb_compressed_size,
        },
        .iconRegular = {
            .data = far_compressed_data,
            .size = far_compressed_size,
        },
        .iconSolid = {
            .data = fas_compressed_data,
            .size = fas_compressed_size,
        },
    });

    // Prime the initial workbench view state.

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
        callbacks.notifyResult(completed.result, completed.message);
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

    if (state.modal.content == ModalContent::FlowgraphInfo ||
        state.modal.content == ModalContent::FlowgraphClose) {
        const std::optional<std::string> targetFlowgraph = state.modal.flowgraph.has_value()
            ? state.modal.flowgraph
            : state.interface.focusedFlowgraph;
        if (!targetFlowgraph.has_value() || !state.flowgraph.items.contains(targetFlowgraph.value())) {
            state.modal.content.reset();
            state.modal.flowgraph.reset();
        }
        return;
    }

    if (state.modal.content == ModalContent::RenameBlock &&
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
