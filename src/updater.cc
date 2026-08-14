#include "updater.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include <algorithm>
#include <exception>
#include <mutex>
#include <optional>
#include <regex>
#include <thread>
#include <utility>
#include <vector>

#if defined(JETSTREAM_LOADER_VELOPACK_AVAILABLE)
#include <Velopack.hpp>
#endif

namespace Jetstream {

namespace {

std::string NormalizeGithubReleaseNotes(std::string notes) {
    static const std::regex pullRequestUrl(
        R"((https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/([0-9]+)))");
    static const std::regex compareUrl(
        R"((https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/compare/([^ \t\r\n)]+)))");
    static const std::regex mention(R"((^|[ \t\r\n])@([A-Za-z0-9-]+))");

    notes = std::regex_replace(notes, pullRequestUrl, "[#$2]($1)");
    notes = std::regex_replace(notes, compareUrl, "[$2]($1)");
    return std::regex_replace(notes, mention, "$1[@$2](https://github.com/$2)");
}

#if defined(JETSTREAM_LOADER_VELOPACK_AVAILABLE)
std::vector<std::string> restartArguments;
#endif

}  // namespace

struct Updater::Impl {
    mutable std::mutex mutex;
    std::mutex workerMutex;
    Snapshot state;
    bool busy = false;
    bool active = true;
    std::thread worker;

#if defined(JETSTREAM_LOADER_VELOPACK_AVAILABLE)
    std::optional<Velopack::UpdateInfo> update;
    std::optional<Velopack::VelopackAsset> pending;

    static std::unique_ptr<Velopack::UpdateManager> CreateManager() {
        std::string source;
        if (Platform::EnvironmentVariable("CYBERETHER_UPDATE_SOURCE", source) == Result::SUCCESS &&
            !source.empty()) {
            return std::make_unique<Velopack::UpdateManager>(source);
        }

        auto github = std::make_unique<Velopack::GithubSource>("https://github.com/luigifcruz/CyberEther");
        return std::make_unique<Velopack::UpdateManager>(std::move(github));
    }

    static void DownloadProgress(void* userData, size_t progress) {
        auto* impl = static_cast<Impl*>(userData);
        std::lock_guard lock(impl->mutex);
        impl->state.progress = static_cast<F32>(std::min<size_t>(progress, 100)) / 100.0f;
    }
#endif

    void fail(const std::string& operation, const std::string& message) {
        std::lock_guard lock(mutex);
        busy = false;
        state.checking = false;
        state.downloading = false;
        state.applying = false;
        state.upToDate = false;
        state.failed = true;
        state.message = operation + " failed: " + message;
    }
};

Updater::Updater() : pimpl(std::make_shared<Impl>()) {
#if defined(JETSTREAM_LOADER_VELOPACK_AVAILABLE)
    try {
        auto manager = Impl::CreateManager();
        pimpl->state.currentVersion = manager->GetCurrentVersion();
        pimpl->state.supported = !pimpl->state.currentVersion.empty();
        if (!pimpl->state.supported) {
            pimpl->state.message = "Automatic updates are available only in official packages.";
            return;
        }

        auto pending = manager->UpdatePendingRestart();
        if (pending.has_value()) {
            pimpl->pending = std::move(pending);
            pimpl->state.ready = true;
            pimpl->state.progress = 1.0f;
            pimpl->state.version = pimpl->pending->Version;
            pimpl->state.releaseNotes = NormalizeGithubReleaseNotes(pimpl->pending->NotesMarkdown);
            pimpl->state.message = "A downloaded update is ready to install.";
        }
    } catch (...) {
        pimpl->state.message = "Automatic updates are available only in official packages.";
    }
#else
    pimpl->state.message = "Automatic updates are available only in packaged desktop builds.";
#endif
}

Updater::~Updater() {
    shutdown();
}

void Updater::Initialize(int argc, char* argv[]) {
#if defined(JETSTREAM_LOADER_VELOPACK_AVAILABLE)
    restartArguments.assign(argv + 1, argv + argc);
    Velopack::VelopackApp::Build()
        .SetAutoApplyOnStartup(false)
        .Run();
#else
    (void)argc;
    (void)argv;
#endif
}

void Updater::start() {
    std::lock_guard workerLock(pimpl->workerMutex);
    std::lock_guard lock(pimpl->mutex);
    pimpl->active = true;
}

Updater::Snapshot Updater::snapshot() const {
    std::lock_guard lock(pimpl->mutex);
    return pimpl->state;
}

void Updater::check() {
#if defined(JETSTREAM_LOADER_VELOPACK_AVAILABLE)
    auto impl = pimpl;
    std::lock_guard workerLock(impl->workerMutex);
    {
        std::lock_guard lock(impl->mutex);
        if (!impl->active || !impl->state.supported || impl->busy ||
            impl->state.available || impl->state.ready) {
            return;
        }

        impl->busy = true;
        impl->update.reset();
        impl->state.upToDate = false;
        impl->state.checking = true;
        impl->state.available = false;
        impl->state.downloading = false;
        impl->state.ready = false;
        impl->state.applying = false;
        impl->state.failed = false;
        impl->state.progress = 0.0f;
        impl->state.version.clear();
        impl->state.releaseNotes.clear();
        impl->state.message = "Checking for updates...";
    }

    if (impl->worker.joinable()) {
        impl->worker.join();
    }
    impl->worker = std::thread([impl]() {
        try {
            auto manager = Impl::CreateManager();
            auto update = manager->CheckForUpdates();

            std::lock_guard lock(impl->mutex);
            impl->busy = false;
            impl->state.checking = false;
            if (update.has_value()) {
                impl->state.available = true;
                impl->state.version = update->TargetFullRelease.Version;
                impl->state.releaseNotes = NormalizeGithubReleaseNotes(update->TargetFullRelease.NotesMarkdown);
                impl->state.message = "A new CyberEther release is available.";
                impl->update = std::move(update);
            } else {
                impl->state.upToDate = true;
                impl->state.message = "CyberEther is up to date.";
            }
        } catch (const std::exception& error) {
            JST_WARN("[UPDATER] Update check failed: {}", error.what());
            impl->fail("Update check", error.what());
        } catch (...) {
            JST_WARN("[UPDATER] Update check failed with an unknown error.");
            impl->fail("Update check", "unknown error");
        }
    });
#endif
}

void Updater::download() {
#if defined(JETSTREAM_LOADER_VELOPACK_AVAILABLE)
    auto impl = pimpl;
    std::optional<Velopack::UpdateInfo> update;

    std::lock_guard workerLock(impl->workerMutex);
    {
        std::lock_guard lock(impl->mutex);
        if (!impl->active || impl->busy || !impl->state.available ||
            impl->state.ready || !impl->update.has_value()) {
            return;
        }

        impl->busy = true;
        impl->state.downloading = true;
        impl->state.ready = false;
        impl->state.failed = false;
        impl->state.progress = 0.0f;
        impl->state.message = "Downloading the update...";
        update = impl->update;
    }

    if (impl->worker.joinable()) {
        impl->worker.join();
    }
    impl->worker = std::thread([impl, update = std::move(update)]() {
        try {
            auto manager = Impl::CreateManager();
            manager->DownloadUpdates(*update, Impl::DownloadProgress, impl.get());

            std::lock_guard lock(impl->mutex);
            impl->busy = false;
            impl->state.available = false;
            impl->state.downloading = false;
            impl->state.ready = true;
            impl->state.progress = 1.0f;
            impl->state.message = "The update is ready to install.";
        } catch (const std::exception& error) {
            JST_WARN("[UPDATER] Update download failed: {}", error.what());
            impl->fail("Update download", error.what());
        } catch (...) {
            JST_WARN("[UPDATER] Update download failed with an unknown error.");
            impl->fail("Update download", "unknown error");
        }
    });
#endif
}

bool Updater::apply(bool restart) {
#if defined(JETSTREAM_LOADER_VELOPACK_AVAILABLE)
    std::optional<Velopack::UpdateInfo> update;
    std::optional<Velopack::VelopackAsset> pending;
    std::lock_guard workerLock(pimpl->workerMutex);
    {
        std::lock_guard lock(pimpl->mutex);
        if (!pimpl->active || pimpl->busy || !pimpl->state.ready) {
            return false;
        }

        pimpl->busy = true;
        pimpl->state.applying = true;
        pimpl->state.failed = false;
        pimpl->state.message = restart
            ? "Restarting to install the update..."
            : "Preparing to install the update...";
        update = pimpl->update;
        pending = pimpl->pending;
    }

    try {
        auto manager = Impl::CreateManager();
        if (pending.has_value()) {
            manager->WaitExitThenApplyUpdates(
                *pending,
                false,
                restart,
                restart ? restartArguments : std::vector<std::string>{});
        } else {
            manager->WaitExitThenApplyUpdates(
                *update,
                false,
                restart,
                restart ? restartArguments : std::vector<std::string>{});
        }
        return true;
    } catch (const std::exception& error) {
        JST_WARN("[UPDATER] Failed to start the update helper: {}", error.what());
        pimpl->fail("Starting the update", error.what());
    } catch (...) {
        JST_WARN("[UPDATER] Failed to start the update helper with an unknown error.");
        pimpl->fail("Starting the update", "unknown error");
    }
#endif

    return false;
}

void Updater::dismiss() {
    std::lock_guard lock(pimpl->mutex);
    if (pimpl->busy || !pimpl->state.available || pimpl->state.ready) {
        return;
    }

    pimpl->state.available = false;
    pimpl->state.upToDate = false;
    pimpl->state.failed = false;
    pimpl->state.version.clear();
    pimpl->state.releaseNotes.clear();
    pimpl->state.message.clear();
#if defined(JETSTREAM_LOADER_VELOPACK_AVAILABLE)
    pimpl->update.reset();
#endif
}

void Updater::shutdown() {
    std::lock_guard workerLock(pimpl->workerMutex);
    {
        std::lock_guard lock(pimpl->mutex);
        pimpl->active = false;
    }

    if (pimpl->worker.joinable()) {
        pimpl->worker.join();
    }
}

}  // namespace Jetstream
