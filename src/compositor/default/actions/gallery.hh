#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_GALLERY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_GALLERY_HH

#include "../model/state.hh"
#include "../support/gallery.hh"

#include "jetstream/platform.hh"
#include "jetstream/plugin.hh"
#include "jetstream/registry.hh"
#include "jetstream/settings.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace Jetstream {

struct GalleryActions {
    using Clock = std::chrono::steady_clock;
    using Flowgraphs = std::vector<Registry::FlowgraphRegistration>;
    using Fetched = DefaultCompositorState::RegistryState::GalleryFetch::Fetched;
    using Fetcher = std::function<Result(const Gallery::Query&, const std::function<bool()>&, Flowgraphs&)>;
    using Loader = std::function<Result(Flowgraphs&)>;

    static constexpr auto RetryInterval = std::chrono::seconds(30);
    static constexpr auto RefreshInterval = std::chrono::hours(1);
    static constexpr auto ProgressMinimum = std::chrono::seconds(1);
    static constexpr const char* PublishedPrefix = "gallery/";

    DefaultCompositorState& state;
    Fetcher fetcher;
    Loader loader;
    bool restored = false;

    explicit GalleryActions(DefaultCompositorState& state,
                            Fetcher fetcher = &Gallery::Fetch,
                            Loader loader = &Gallery::Load)
        : state(state), fetcher(std::move(fetcher)), loader(std::move(loader)) {}

    void reconcile(Clock::time_point now = Clock::now()) {
        auto& gallery = state.registry.gallery;
        if (!restored) {
            restored = true;
            Flowgraphs cached;
            if (loader && loader(cached) == Result::SUCCESS && !cached.empty()) {
                gallery.flowgraphs = std::move(cached);
                gallery.cached = true;
                state.registry.stale = true;
            }
        }
        if (gallery.loading) {
            if (gallery.future.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
                return;
            }
            if (!gallery.attempted && now < gallery.started + ProgressMinimum) {
                return;
            }
            collect(now);
            return;
        }
        if (now < gallery.next) {
            return;
        }
        gallery.loading = true;
        gallery.started = now;
        gallery.cancel = std::make_shared<std::atomic<bool>>(false);
        gallery.future = std::async(std::launch::async,
                                    [fetcher = fetcher, query = Gallery::Current(), cancel = gallery.cancel]()
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

    static Flowgraphs Catalog(const Flowgraphs& published,
                              Flowgraphs bundled,
                              const std::vector<Plugin::Info>& plugins) {
        std::set<std::string> examples;
        for (const auto& plugin : plugins) {
            examples.insert(plugin.examples.begin(), plugin.examples.end());
        }
        Flowgraphs catalog = published;
        for (auto& entry : catalog) {
            if (examples.contains(entry.key)) {
                entry.key = PublishedPrefix + entry.key;
            }
        }
        for (auto& entry : bundled) {
            const bool replaced = std::any_of(catalog.begin(), catalog.end(), [&](const auto& item) {
                return item.key == entry.key;
            });
            if (!replaced) {
                catalog.push_back(std::move(entry));
            }
        }
        return catalog;
    }

    void shutdown() {
        auto& gallery = state.registry.gallery;
        if (!gallery.loading) {
            return;
        }
        gallery.cancel->store(true);
        gallery.future.wait();
        gallery.loading = false;
        (void)gallery.future.get();
    }

    void refreshRegistry() {
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
        const auto plugins = Plugin::List();
        registry.flowgraphs = Catalog(registry.gallery.flowgraphs, Registry::ListAvailableFlowgraphs(), plugins);
        registry.registered.clear();

        Settings settings;
        if (Settings::Get(settings) != Result::SUCCESS) {
            return;
        }
        registry.stale = false;
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

 private:
    static std::string NormalizePluginPath(const std::string& path) {
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

    void collect(Clock::time_point now) {
        auto& gallery = state.registry.gallery;
        gallery.loading = false;
        gallery.attempted = true;
        auto fetched = gallery.future.get();
        gallery.failed = !fetched.has_value();
        if (!fetched.has_value()) {
            gallery.next = now + RetryInterval;
            return;
        }
        gallery.next = now + (fetched->complete ? RefreshInterval : RetryInterval);
        gallery.flowgraphs = std::move(fetched->flowgraphs);
        state.registry.stale = true;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_GALLERY_HH
