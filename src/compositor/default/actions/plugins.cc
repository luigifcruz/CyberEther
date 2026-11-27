#include "plugins.hh"

#include <algorithm>
#include <mutex>

namespace Jetstream {

namespace {

using Outcome = DefaultCompositorState::PluginState::Task::Outcome;

template<typename T>
bool Ready(const std::future<T>& future) {
    return future.valid() && future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

}  // namespace

PluginActions::PluginActions(DefaultCompositorState& state,
                             DefaultCompositorCallbacks& callbacks,
                             Plugin::Gallery::Source source)
    : state(state), callbacks(callbacks), source(std::move(source)) {}

bool PluginActions::ValidUpdatePolicy(const std::string& value) {
    return value == "prompt" || value == "auto" || value == "manual";
}

std::vector<std::string> PluginActions::Missing(const std::vector<char>& blob) {
    std::vector<std::string> plugins;
    if (Flowgraph::ReadPlugins(blob, plugins) != Result::SUCCESS) {
        return {};
    }

    std::vector<std::string> missing;
    for (const auto& id : plugins) {
        if (!Plugin::Gallery::ValidId(id) || std::find(missing.begin(), missing.end(), id) != missing.end()) {
            continue;
        }
        if (Plugin::Path(id).empty()) {
            missing.push_back(id);
        }
    }
    return missing;
}

std::vector<std::string> PluginActions::MissingInFile(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(Platform::PathFromUtf8(path), ec)) {
        return {};
    }

    std::vector<char> blob;
    try {
        std::ifstream file(Platform::PathFromUtf8(path), std::ios::binary);
        blob.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    } catch (...) {
        return {};
    }
    return Missing(blob);
}

void PluginActions::Request(DefaultCompositorState& state,
                            std::vector<std::string> missing,
                            std::string path,
                            std::vector<char> blob) {
    state.plugins.missing = std::move(missing);
    state.plugins.path = std::move(path);
    state.plugins.blob = state.plugins.path.empty() ? std::move(blob) : std::vector<char>{};
    state.modal = {.content = ModalContent::MissingPlugins};
}

void PluginActions::reconcile(Clock::time_point now) {
    reconcileCheck();
    reconcileUpdate();
    reconcileCatalog(now);
    reconcileInstall();
    reconcilePrompt();
}

void PluginActions::shutdown() {
    state.plugins.cancel->store(true);
    for (auto* task : {&state.plugins.install, &state.plugins.update}) {
        if (task->future.valid()) {
            task->future.wait();
        }
        task->loading = false;
    }
    for (auto* catalog : {&state.plugins.catalog, &state.plugins.updates}) {
        if (catalog->future.valid()) {
            catalog->future.wait();
        }
        catalog->loading = false;
    }
}

void PluginActions::reconcileCheck() {
    auto& updates = state.plugins.updates;
    if (!state.plugins.checked) {
        state.plugins.checked = true;
        updates.loading = true;
        updates.future = std::async(std::launch::async, [source = source, cancel = state.plugins.cancel]()
                                        -> std::optional<std::vector<Plugin::Gallery::Entry>> {
            Plugin::Gallery::Entries entries;
            const auto cancelled = [cancel] {
                return cancel->load();
            };
            try {
                if (Plugin::Gallery::Updates(source, entries, cancelled) != Result::SUCCESS) {
                    return std::nullopt;
                }
            } catch (...) {
                return std::nullopt;
            }
            return entries;
        });
        return;
    }

    if (!updates.loading || !Ready(updates.future) || state.benchmark.running) {
        return;
    }
    updates.loading = false;

    auto entries = updates.future.get();
    updates.failed = !entries.has_value();
    if (!entries.has_value() || entries->empty()) {
        return;
    }
    updates.entries = std::move(*entries);

    if (state.plugins.updatePolicy == "auto") {
        std::vector<std::string> ids;
        for (const auto& entry : updates.entries) {
            ids.push_back(entry.id);
        }
        (void)handle(MailUpdateGalleryPlugins{.ids = std::move(ids)});
    } else if (state.plugins.updatePolicy == "prompt") {
        state.plugins.prompt = true;
    }
}

void PluginActions::reconcilePrompt() {
    if (state.modal.content == ModalContent::PluginUpdates && !state.registry.stale &&
        state.pluginUpdates().empty()) {
        state.modal = {};
    }
    if (state.plugins.updatePolicy != "prompt") {
        state.plugins.prompt = false;
    }
    if (!state.plugins.prompt || state.modal.content.has_value() || state.interface.closingFlowgraph.has_value() ||
        state.filePicker.active || state.registry.stale || state.plugins.update.loading ||
        state.plugins.install.loading || state.benchmark.running) {
        return;
    }
    state.plugins.prompt = false;
    if (!state.pluginUpdates().empty()) {
        state.modal = {.content = ModalContent::PluginUpdates};
    }
}

void PluginActions::reconcileUpdate() {
    auto& update = state.plugins.update;
    if (!update.loading || !Ready(update.future) || state.benchmark.running) {
        return;
    }
    update.loading = false;

    const auto outcome = update.future.get();
    const auto requested = std::move(update.ids);
    update.ids.clear();

    const auto& settled = outcome.result == Result::SUCCESS ? requested : outcome.ids;
    std::erase_if(state.plugins.updates.entries, [&](const auto& entry) {
        return std::find(settled.begin(), settled.end(), entry.id) != settled.end();
    });

    if (!outcome.ids.empty()) {
        (void)Plugin::LoadInstalled();
        state.registry.stale = true;
        callbacks.notify(Sakura::ToastType::Info,
                         5000,
                         jst::fmt::format("Updated plugins from the gallery: {}. New versions of loaded plugins "
                                          "take effect the next time CyberEther starts.",
                                          Text::Join(outcome.ids, ", ")));
    }

    if (outcome.result != Result::SUCCESS) {
        const auto message = outcome.error.empty() ? "Failed to update plugins from the gallery." : outcome.error;
        callbacks.notifyResult(Result::ERROR, message);
    }
}

void PluginActions::reconcileCatalog(Clock::time_point now) {
    auto& catalog = state.plugins.catalog;
    if (catalog.loading) {
        if (!Ready(catalog.future)) {
            return;
        }
        catalog.loading = false;
        auto entries = catalog.future.get();
        catalog.failed = !entries.has_value();
        if (entries.has_value()) {
            catalog.entries = std::move(*entries);
        }
        catalog.next = now + (catalog.failed ? RetryInterval : RefreshInterval);
        return;
    }

    if (now < catalog.next) {
        return;
    }

    catalog.loading = true;
    catalog.future = std::async(std::launch::async, [source = source, cancel = state.plugins.cancel]()
                                    -> std::optional<std::vector<Plugin::Gallery::Entry>> {
        Plugin::Gallery::Entries entries;
        const auto cancelled = [cancel] {
            return cancel->load();
        };
        try {
            if (Plugin::Gallery::Fetch(source, Plugin::Gallery::Current(), cancelled, entries) != Result::SUCCESS) {
                return std::nullopt;
            }
        } catch (...) {
            return std::nullopt;
        }
        return entries;
    });
}

void PluginActions::reconcileInstall() {
    auto& install = state.plugins.install;
    if (!install.loading || !Ready(install.future) || state.benchmark.running) {
        return;
    }
    install.loading = false;

    const auto outcome = install.future.get();
    auto path = std::move(install.path);
    auto blob = std::move(install.blob);
    install.ids.clear();
    install.path.clear();
    install.blob.clear();

    if (!outcome.ids.empty()) {
        if (Plugin::LoadInstalled() != Result::SUCCESS) {
            callbacks.notifyResult(Result::ERROR, "");
        }
        state.registry.stale = true;
        state.plugins.catalog.next = {};
        std::erase_if(state.plugins.updates.entries, [&](const auto& entry) {
            return std::find(outcome.ids.begin(), outcome.ids.end(), entry.id) != outcome.ids.end();
        });
    }

    if (outcome.result != Result::SUCCESS) {
        const auto message = outcome.error.empty() ? "Failed to install the plugin from the gallery." : outcome.error;
        callbacks.notifyResult(Result::ERROR, message);
        return;
    }

    std::vector<std::string> unloaded;
    for (const auto& id : outcome.ids) {
        if (Plugin::Path(id).empty()) {
            unloaded.push_back(id);
        }
    }
    if (!unloaded.empty()) {
        callbacks.notify(Sakura::ToastType::Error,
                         5000,
                         jst::fmt::format("Installed {} but could not load {}. Check the logs for details.",
                                          Text::Join(unloaded, ", "),
                                          unloaded.size() == 1 ? "it" : "them"));
        return;
    }

    if (!path.empty()) {
        callbacks.enqueueMail(MailOpenFlowgraphPath{.path = std::move(path)});
    } else if (!blob.empty()) {
        callbacks.enqueueMail(MailOpenFlowgraphBlob{.blob = std::move(blob)});
    }

    callbacks.notify(Sakura::ToastType::Success,
                     3000,
                     outcome.ids.size() == 1 ? "Plugin installed." : "Plugins installed.");
}

Result PluginActions::handle(const MailOpenPluginDetails& msg) {
    auto& catalog = state.plugins.catalog;
    const bool listed = std::ranges::any_of(catalog.entries, [&](const auto& entry) { return entry.id == msg.id; });
    if (!listed && !catalog.loading) {
        catalog.next = {};
    }
    state.plugins.details = msg.id;
    return Result::SUCCESS;
}

Result PluginActions::handle(const MailClosePluginDetails&) {
    state.plugins.details.clear();
    return Result::SUCCESS;
}

Result PluginActions::handle(const MailInstallGalleryPlugins& msg) {
    auto& install = state.plugins.install;
    if (install.loading || state.plugins.update.loading) {
        callbacks.notify(Sakura::ToastType::Info, 3000, "Another plugin is still being installed.");
        return Result::SUCCESS;
    }
    std::vector<std::string> ids;
    for (const auto& id : msg.ids) {
        if (!msg.reopen || Plugin::Path(id).empty()) {
            ids.push_back(id);
        }
    }

    if (state.modal.content == ModalContent::MissingPlugins) {
        state.modal = {};
    }

    if (ids.empty()) {
        if (msg.reopen) {
            if (!state.plugins.path.empty()) {
                callbacks.enqueueMail(MailOpenFlowgraphPath{.path = std::move(state.plugins.path)});
            } else if (!state.plugins.blob.empty()) {
                callbacks.enqueueMail(MailOpenFlowgraphBlob{.blob = std::move(state.plugins.blob)});
            }
            state.plugins.missing.clear();
            state.plugins.path.clear();
            state.plugins.blob.clear();
        }
        return Result::SUCCESS;
    }

    install.loading = true;
    install.ids = ids;
    if (msg.reopen) {
        install.path = std::move(state.plugins.path);
        install.blob = std::move(state.plugins.blob);
        state.plugins.missing.clear();
        state.plugins.path.clear();
        state.plugins.blob.clear();
    }
    install.future = std::async(std::launch::async,
                                [source = source, ids = std::move(ids), cancel = state.plugins.cancel]() {
        Outcome outcome;
        const auto cancelled = [cancel] {
            return cancel->load();
        };
        for (const auto& id : ids) {
            Result result = Result::ERROR;
            try {
                result = Plugin::Gallery::Install(source, id, cancelled);
            } catch (...) {
                result = Result::ERROR;
            }
            if (result != Result::SUCCESS) {
                outcome.result = Result::ERROR;
                std::lock_guard<std::mutex> lock(_JST_LOG_MUTEX());
                outcome.error = JST_LOG_LAST_ERROR();
                return outcome;
            }
            outcome.ids.push_back(id);
        }
        return outcome;
    });

    return Result::SUCCESS;
}

Result PluginActions::handle(const MailUpdateGalleryPlugins& msg) {
    auto& update = state.plugins.update;
    if (update.loading || state.plugins.install.loading) {
        callbacks.notify(Sakura::ToastType::Info, 3000, "Plugin updates are still being installed.");
        return Result::SUCCESS;
    }

    if (state.modal.content == ModalContent::PluginUpdates) {
        state.modal = {};
    }

    if (msg.ids.empty()) {
        return Result::SUCCESS;
    }

    update.loading = true;
    update.ids = msg.ids;
    update.future = std::async(std::launch::async, [source = source, ids = msg.ids, cancel = state.plugins.cancel]() {
        Outcome outcome;
        const auto cancelled = [cancel] {
            return cancel->load();
        };
        try {
            outcome.result = Plugin::Gallery::Update(source, ids, outcome.ids, cancelled);
        } catch (...) {
            outcome.result = Result::ERROR;
        }
        if (outcome.result != Result::SUCCESS) {
            std::lock_guard<std::mutex> lock(_JST_LOG_MUTEX());
            outcome.error = JST_LOG_LAST_ERROR();
        }
        return outcome;
    });

    return Result::SUCCESS;
}

Result PluginActions::handle(const MailSetPluginUpdatePolicy& msg) {
    if (!ValidUpdatePolicy(msg.value)) {
        JST_WARN("[COMPOSITOR_IMPL_DEFAULT] Ignoring invalid plugin update policy '{}'.", msg.value);
        return Result::SUCCESS;
    }

    state.plugins.updatePolicy = msg.value;

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.plugins.updatePolicy = msg.value;
    }));

    return Result::SUCCESS;
}

}  // namespace Jetstream
