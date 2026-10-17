#include "base.hh"

#include "render/sakura/runtime.hh"

#include "jetstream/render/sakura/clipboard.hh"
#include "jetstream/render/sakura/toast.hh"

#include "actions/base.hh"
#include "presenters/base.hh"
#include "support/themes.hh"

#include "resources/fonts/compressed_jbmm.hh"
#include "resources/fonts/compressed_jbmb.hh"
#include "resources/fonts/compressed_fa.hh"
#include "resources/fonts/compressed_sg.hh"

#include "jetstream/platform.hh"
#include "jetstream/logger.hh"
#include "jetstream/instance_remote.hh"
#include "jetstream/settings.hh"

#include <chrono>
#include <deque>
#include <functional>
#include <future>
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
        .enqueueCommandWithCompletion = [this](std::function<Result()> fn, bool silent,
                                               std::function<void(Result)> onComplete) {
            Compositor::Impl::enqueue(std::move(fn), silent, std::move(onComplete));
        },
        .enqueueCommandWithMail = [this](std::function<Result()> fn, bool silent, Mail mail) {
            Compositor::Impl::enqueue(std::move(fn), silent, [this, mail = std::move(mail)](const Result result) {
                if (result == Result::SUCCESS) {
                    enqueue(Mail(mail));
                }
            });
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
        .requestFile = [this](FilePickerRequest request) {
            return actions.requestFile(std::move(request));
        },
        .layout = [this](const std::string& flowgraph, Agent::Tools::Layout layout) {
            agent.executor().layout(flowgraph, std::move(layout));
        },
        .checkForUpdates = [this]() {
            updater.check();
        },
        .downloadUpdate = [this]() {
            updater.download();
        },
        .applyUpdate = [this]() {
            return updater.apply();
        },
        .dismissUpdate = [this]() {
            updater.dismiss();
        },
        .submitFeedback = [this](const std::string& text) {
            feedback.submit(text);
        },
    },
    agent(state, callbacks),
    actions(state, callbacks, agent),
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

    state.sakura.themeKey = ResolveThemeKey(settings.interface.themeKey);
    state.sakura.colorMap = themes.at(state.sakura.themeKey);

    // Restore graphics preferences.

    state.graphics.device = settings.graphics.device;
    state.graphics.scale = settings.graphics.scale;
    state.graphics.framerate = settings.graphics.framerate;

    // Restore interface preferences.

    state.interface.infoPanelEnabled = settings.interface.infoPanelEnabled;
    state.interface.backgroundParticles = settings.interface.backgroundParticles;

    // Restore intelligence preferences.

    state.agent.captureSupported = render && render->supportsCapture();

    state.agent.settings = settings.intelligence;
#ifdef JST_OS_BROWSER
    state.agent.settings.enabled = false;
#endif

    // Restore debug preferences.

    state.debug.logLevel = settings.developer.logLevel;
    state.debug.latencyEnabled = settings.developer.latencyEnabled;
    state.debug.timingEnabled = settings.developer.timingEnabled;
    JST_LOG_SET_DEBUG_LEVEL(state.debug.logLevel);

    // Restore remote streaming preferences.

    state.remote.config.broker.address = settings.remote.broker.address;
    state.remote.config.broker.publicUrl = settings.remote.broker.publicUrl;
    state.remote.config.broker.certificate = settings.remote.broker.certificate;
    state.remote.config.broker.key = settings.remote.broker.key;
    if (settings.remote.broker.port <= 65535) {
        state.remote.config.broker.port = static_cast<U16>(settings.remote.broker.port);
    } else {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Invalid saved broker port {}. Using default.",
                 settings.remote.broker.port);
    }
    state.remote.config.access.autoApprove = settings.remote.access.autoApprove;
    state.remote.config.video.framerate = static_cast<U32>(settings.remote.video.framerate);

    // Restore runtime preferences.

    state.runtime.pythonPath = settings.runtime.python.path;
    state.runtime.dependencyPolicy = settings.runtime.dependencyPolicy;
    if (state.runtime.dependencyPolicy != "prompt" &&
        state.runtime.dependencyPolicy != "allow" &&
        state.runtime.dependencyPolicy != "deny") {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Invalid saved runtime dependency policy '{}'. Using default.",
                 state.runtime.dependencyPolicy);
        state.runtime.dependencyPolicy = "prompt";
    }
    state.runtime.pythonCandidates = PythonRuntimeContext::DiscoverRuntimes();
    state.runtime.pythonValidation = PythonRuntimeContext::ValidateRuntimePath(state.runtime.pythonPath);
    state.runtime.initialPythonValidation = state.runtime.pythonValidation;

    try {
        state.remote.config.video.codec = StringToRemoteCodec(settings.remote.video.codec);
    } catch (const Result&) {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Invalid saved remote codec '{}'. Using default.",
                 settings.remote.video.codec);
    }

    try {
        state.remote.config.video.encoder = StringToRemoteEncoder(settings.remote.video.encoder);
    } catch (const Result&) {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Invalid saved remote encoder '{}'. Using default.",
                 settings.remote.video.encoder);
    }

    try {
        state.remote.config.media.transport = StringToRemoteTransport(settings.remote.media.transport);
    } catch (const Result&) {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Invalid saved remote transport '{}'. Using default.",
                 settings.remote.media.transport);
    }

    if (IsRemotePortRangeValid(settings.remote.media.minPort, settings.remote.media.maxPort)) {
        state.remote.config.media.minPort = static_cast<U16>(settings.remote.media.minPort);
        state.remote.config.media.maxPort = static_cast<U16>(settings.remote.media.maxPort);
    } else {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Invalid saved remote port range {}-{}. Using default.",
                 settings.remote.media.minPort, settings.remote.media.maxPort);
    }

    state.remote.config.media.address = settings.remote.media.address;

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
        .display = {
            .data = sg_compressed_data,
            .size = sg_compressed_size,
        },
    });

    // Prime the initial workbench view state.

    workbench.update(presenters.build());

    updater.start();
    updater.check();

    return Result::SUCCESS;
}

Result DefaultCompositor::destroy() {
    JST_INFO("[COMPOSITOR_IMPL_DEFAULT] Destroying compositor.");

    actions.cancelFilePicker();
    agent.shutdown();
    actions.shutdownAgent();
    updater.shutdown();
    feedback.shutdown();
    if (Settings::Flush() != Result::SUCCESS) {
        JST_ERROR("[COMPOSITOR_IMPL_DEFAULT] Failed to save settings during shutdown. Continuing cleanup.");
    }

    return Result::SUCCESS;
}

Result DefaultCompositor::poll() {
    // Route files dropped onto the window into the mail queue.

    if (state.system.viewport) {
        for (auto& path : state.system.viewport->takeDroppedFiles()) {
            enqueue(MailDropFlowgraphPath{std::move(path)});
        }
    }

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

    agent.poll();
    while (const auto error = Settings::TakePersistenceError()) {
        callbacks.notify(Sakura::ToastType::Error, 5000, *error);
    }

    // Surface completed async command results as notifications.

    Command completed;
    while (dequeue(completed)) {
        if (completed.onComplete) {
            completed.onComplete(completed.result);
        }
        if (completed.silent) {
            continue;
        }
        callbacks.notifyResult(completed.result, completed.message);
    }

    // Update the state snapshots used by presenters.

    updateWorkbenchState();
    updateFlowgraphBlockState();
    updateFilePendingState();
    updateDependencyState();
    updateBenchmarkState();
    updateRemoteState();
    updateUpdaterState();
    actions.reconcileFilePicker();
    actions.restoreStacks();
    actions.refreshLiveMarkdown();
    actions.reconcileAgent();
    updateFeedbackState();

    // Build view configs while flowgraph access is confined to poll.

    workbench.update(presenters.build());
    state.flowgraph.blocks.clear();

    return Result::SUCCESS;
}

Result DefaultCompositor::present() {
    state.sakura.runtime.update({
        .palette = &state.sakura.colorMap,
        .render = state.system.render.get(),
    });

    const auto& ctx = state.sakura.runtime.context();
    workbench.render(ctx);

    return Result::SUCCESS;
}

void DefaultCompositor::updateWorkbenchState() {
    if (state.interface.pendingFocusedFlowgraph.has_value() &&
        state.flowgraph.items.contains(*state.interface.pendingFocusedFlowgraph)) {
        state.interface.focusedFlowgraph = state.interface.pendingFocusedFlowgraph;
        state.interface.pendingFocusedFlowgraph.reset();
    }

    if (state.flowgraph.items.empty()) {
        state.interface.focusedFlowgraph.reset();
        state.interface.flowgraphMetadataVisible = false;
        state.interface.flowgraphMetadataSearch.clear();
        state.interface.flowgraphEnvironmentVisible = false;
        state.interface.flowgraphEnvironmentSearch.clear();
    } else if (state.interface.focusedFlowgraph.has_value() &&
               !state.flowgraph.items.contains(state.interface.focusedFlowgraph.value())) {
        state.interface.focusedFlowgraph.reset();
        state.interface.flowgraphMetadataVisible = false;
        state.interface.flowgraphMetadataSearch.clear();
        state.interface.flowgraphEnvironmentVisible = false;
        state.interface.flowgraphEnvironmentSearch.clear();
    }

    if (!state.interface.focusedFlowgraph.has_value() &&
        !state.interface.pendingFocusedFlowgraph.has_value() &&
        state.flowgraph.items.size() == 1) {
        state.interface.focusedFlowgraph = state.flowgraph.items.begin()->first;
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

    if (state.modal.content == ModalContent::RenameBlock) {
        const std::optional<std::string> targetFlowgraph = state.modal.flowgraph.has_value()
            ? state.modal.flowgraph
            : state.interface.focusedFlowgraph;
        if (!targetFlowgraph.has_value() ||
            !state.flowgraph.items.contains(targetFlowgraph.value()) ||
            !state.modal.renameBlockOldName.has_value() ||
            !state.flowgraph.items.at(targetFlowgraph.value())->view().has(state.modal.renameBlockOldName.value())) {
            state.modal.content.reset();
            state.modal.flowgraph.reset();
            state.modal.renameBlockOldName.reset();
        }
    }
}

void DefaultCompositor::updateFlowgraphBlockState() {
    state.flowgraph.blocks.clear();
    for (const auto& [flowgraphId, flowgraph] : state.flowgraph.items) {
        auto& blocks = state.flowgraph.blocks[flowgraphId];
        std::vector<std::string> names;
        if (!flowgraph || flowgraph->view().keys(names) != Result::SUCCESS) {
            continue;
        }
        blocks.reserve(names.size());
        for (auto& name : names) {
            Flowgraph::View::BlockData data;
            if (flowgraph->view().block(name, data) != Result::SUCCESS) {
                continue;
            }
            blocks.push_back({std::move(name), std::move(data)});
        }
    }
}

void DefaultCompositor::updateFilePendingState() {
#ifdef JST_OS_BROWSER
    state.interface.filePending = Platform::IsFilePending();
#else
    state.interface.filePending = false;
#endif
}

void DefaultCompositor::updateDependencyState() {
    auto snapshot = SnapshotPythonDependencyState();
    state.runtime.dependencyRequest = std::move(snapshot.request);
    state.runtime.dependencyGeneration = snapshot.generation;
}

void DefaultCompositor::updateBenchmarkState() {
    if (state.benchmark.running && state.benchmark.future.valid()) {
        const auto status = state.benchmark.future.wait_for(std::chrono::milliseconds(0));
        if (status == std::future_status::ready) {
            state.benchmark.future.get();
            state.benchmark.running = false;
        }
    }

    if (!state.benchmark.running && state.modal.content != ModalContent::Benchmark) {
        return;
    }

    const U64 current = Benchmark::CurrentCount();
    const U64 total = Benchmark::TotalCount(state.benchmark.selectedModule);
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

void DefaultCompositor::updateUpdaterState() {
    const auto update = updater.snapshot();
    state.update.supported = update.supported;
    state.update.upToDate = update.upToDate;
    state.update.failed = update.failed;
    state.update.checking = update.checking;
    state.update.available = update.available;
    state.update.downloading = update.downloading;
    state.update.ready = update.ready;
    state.update.applying = update.applying;
    state.update.progress = update.progress;
    state.update.version = update.version;
    state.update.releaseNotes = update.releaseNotes;
    state.update.message = update.message;
}

void DefaultCompositor::enqueue(Mail&& mail) {
    pendingMail.emplace_back(std::move(mail));
}

void DefaultCompositor::updateFeedbackState() {
    const bool modalOpen = state.modal.content.has_value() &&
                           state.modal.content.value() == ModalContent::Feedback;

    const auto snapshot = feedback.snapshot();

    switch (snapshot.status) {
        case Feedback::Snapshot::Status::Idle:
            if (state.feedback.status == DefaultCompositorState::FeedbackState::Status::Submitting) {
                state.feedback.status = DefaultCompositorState::FeedbackState::Status::Idle;
            }
            if (!modalOpen &&
                (state.feedback.status != DefaultCompositorState::FeedbackState::Status::Idle ||
                 !state.feedback.text.empty())) {
                state.feedback = {};
                feedback.clear();
            }
            break;
        case Feedback::Snapshot::Status::Submitting:
            state.feedback.status = DefaultCompositorState::FeedbackState::Status::Submitting;
            break;
        case Feedback::Snapshot::Status::Success:
            if (state.feedback.status == DefaultCompositorState::FeedbackState::Status::Submitting) {
                callbacks.notify(Sakura::ToastType::Success, 5000,
                                 "Feedback submitted. Thank you!");
                feedback.clear();
                state.feedback = {};
            }
            break;
        case Feedback::Snapshot::Status::Error:
            if (state.feedback.status == DefaultCompositorState::FeedbackState::Status::Submitting) {
                callbacks.notify(Sakura::ToastType::Error, 5000,
                                 snapshot.message);
                feedback.clear();
                state.feedback = {};
            }
            break;
    }
}

std::shared_ptr<Compositor::Impl> DefaultCompositorFactory() {
    return std::make_shared<DefaultCompositor>();
}

}  // namespace Jetstream
