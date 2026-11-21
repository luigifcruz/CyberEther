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
                            Loader loader = &Gallery::Load);
    void reconcile(Clock::time_point now = Clock::now());
    static Flowgraphs Catalog(const Flowgraphs& published,
                              Flowgraphs bundled,
                              const std::vector<Plugin::Info>& plugins);
    void shutdown();
    void refreshRegistry();

 private:
    static std::string NormalizePluginPath(const std::string& path);
    void collect(Clock::time_point now);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_GALLERY_HH
