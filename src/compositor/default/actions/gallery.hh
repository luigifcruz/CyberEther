#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_GALLERY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_GALLERY_HH

#include "../model/state.hh"
#include "home_gallery_remote.hh"

#include "jetstream/home.hh"
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
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace Jetstream {

struct GalleryActions {
    using Clock = std::chrono::steady_clock;
    using Remote = Home::Gallery::Remote;
    using Flowgraphs = Remote::Flowgraphs;
    using Fetched = DefaultCompositorState::RegistryState::GalleryFetch::Fetched;
    using Fetcher = std::function<Result(const Remote::Query&, const std::function<bool()>&, Flowgraphs&)>;
    using Loader = std::function<Result(Flowgraphs&)>;

    static constexpr auto RetryInterval = std::chrono::seconds(30);
    static constexpr auto RefreshInterval = std::chrono::hours(1);
    static constexpr auto ProgressMinimum = std::chrono::seconds(1);

    DefaultCompositorState& state;
    Home::Gallery& gallery;
    Fetcher fetcher;
    Loader loader;
    bool restored = false;

    GalleryActions(DefaultCompositorState& state,
                   Home::Gallery& gallery,
                   Fetcher fetcher = &Remote::Fetch,
                   Loader loader = &Remote::Load);
    void reconcile(Clock::time_point now = Clock::now());
    void shutdown();
    void refreshRegistry();

 private:
    static std::string NormalizePluginPath(const std::string& path);
    bool publish(const Flowgraphs& flowgraphs);
    void collect(Clock::time_point now);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_GALLERY_HH
