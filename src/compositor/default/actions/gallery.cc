#include "gallery.hh"

namespace Jetstream {

GalleryActions::GalleryActions(DefaultCompositorState& state,
                               Home::Gallery& gallery,
                               Fetcher fetcher,
                               Loader loader)
    : state(state), gallery(gallery), fetcher(std::move(fetcher)), loader(std::move(loader)) {}

void GalleryActions::reconcile(Clock::time_point now) {
    auto& fetch = state.registry.fetch;
    if (!restored) {
        restored = true;
        Flowgraphs cached;
        if (loader && loader(cached) == Result::SUCCESS && !cached.empty() && publish(cached)) {
            fetch.cached = true;
        }
    }
    if (gallery.revision() != state.registry.revision) {
        state.registry.stale = true;
    }
    if (fetch.loading) {
        if (fetch.future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            return;
        }
        if (!fetch.attempted && now < fetch.started + ProgressMinimum) {
            return;
        }
        collect(now);
        return;
    }
    if (now < fetch.next) {
        return;
    }
    fetch.loading = true;
    fetch.started = now;
    fetch.cancel = std::make_shared<std::atomic<bool>>(false);
    fetch.future = std::async(std::launch::async,
                              [fetcher = fetcher, query = Remote::Current(), cancel = fetch.cancel]()
                                  -> std::optional<Fetched> {
        try {
            Flowgraphs flowgraphs;
            const auto cancelled = [cancel] {
                return cancel->load();
            };
            const auto result = fetcher(query, cancelled, flowgraphs);
            if (result != Result::SUCCESS && result != Result::INCOMPLETE) {
                return std::nullopt;
            }
            return Fetched{
                .flowgraphs = std::move(flowgraphs),
                .complete = result == Result::SUCCESS,
            };
        } catch (...) {
            return std::nullopt;
        }
    });
}

void GalleryActions::shutdown() {
    auto& fetch = state.registry.fetch;
    if (!fetch.loading) {
        return;
    }
    fetch.cancel->store(true);
    fetch.future.wait();
    fetch.loading = false;
    (void)fetch.future.get();
}

void GalleryActions::refreshRegistry() {
    auto& registry = state.registry;
    if (!registry.stale) {
        return;
    }
    registry.blocks.clear();
    for (const auto& block : Registry::ListAvailableBlocks()) {
        registry.blocks.push_back({
            .type = block.type,
            .domain = block.domain,
            .title = block.title,
            .summary = block.summary,
        });
    }
    registry.revision = gallery.revision();
    registry.sections = gallery.sections();
    registry.registered.clear();

    Settings settings;
    if (Settings::Get(settings) != Result::SUCCESS) {
        return;
    }
    registry.stale = false;
    const auto plugins = Plugin::List();
    std::map<std::string, const Plugin::Info*> loaded;
    for (const auto& plugin : plugins) {
        loaded[NormalizePluginPath(plugin.path)] = &plugin;
    }
    for (const auto& path : settings.registry.plugins) {
        const auto match = loaded.find(NormalizePluginPath(path));
        std::error_code error;
        DefaultCompositorState::RegistryState::PluginEntry entry{
            .path = path,
            .exists = std::filesystem::exists(Platform::PathFromUtf8(path), error),
        };
        if (match != loaded.end()) {
            entry.info = *match->second;
        }
        registry.registered.push_back(std::move(entry));
    }
}

std::string GalleryActions::NormalizePluginPath(const std::string& path) {
    std::error_code ec;
    const auto inputPath = Platform::PathFromUtf8(path);
    const auto canonical = std::filesystem::weakly_canonical(inputPath, ec);
    if (!ec) {
        return Platform::PathToUtf8(canonical);
    }

    ec.clear();
    const auto absolute = std::filesystem::absolute(inputPath, ec);
    if (!ec) {
        return Platform::PathToUtf8(absolute.lexically_normal());
    }

    return path;
}

bool GalleryActions::publish(const Flowgraphs& flowgraphs) {
    if (gallery.publish(Home::Gallery::Origin, Remote::Sections(flowgraphs)) != Result::SUCCESS) {
        JST_WARN("[GALLERY] Could not publish the fetched flowgraphs.");
        return false;
    }
    state.registry.stale = true;
    return true;
}

void GalleryActions::collect(Clock::time_point now) {
    auto& fetch = state.registry.fetch;
    fetch.loading = false;
    fetch.attempted = true;
    auto fetched = fetch.future.get();
    fetch.failed = !fetched.has_value();
    if (!fetched.has_value()) {
        fetch.next = now + RetryInterval;
        return;
    }
    const bool empty = !fetched->complete && fetched->flowgraphs.empty();
    const bool published = !empty && publish(fetched->flowgraphs);
    fetch.failed = !published;
    fetch.next = now + (published && fetched->complete ? RefreshInterval : RetryInterval);
}

}  // namespace Jetstream
