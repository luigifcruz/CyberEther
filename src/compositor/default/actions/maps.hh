#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_MAPS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_MAPS_HH

#include "persist.hh"
#include "../model/callbacks.hh"
#include "../model/messages.hh"
#include "../model/state.hh"

#include "jetstream/settings.hh"
#include "jetstream/render/components/geomap.hh"
#include "render/components/geomap_tiles.hh"

#include <chrono>
#include <future>
#include <tuple>

namespace Jetstream {

struct MapsActions {
    using Filter = std::tuple<MailSetMapsHighResolution,
                              MailSetMapsRendering,
                              MailClearMapsCache>;
    using CacheResult = DefaultCompositorState::MapsState::CacheResult;
    using Tiles = Render::Components::GeoMapTiles;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    MapsActions(DefaultCompositorState& state,
                DefaultCompositorCallbacks& callbacks) :
        state(state),
        callbacks(callbacks) {}

    Result handle(const MailSetMapsHighResolution& msg) {
        state.maps.highResolution = msg.value;
        Tiles::SetEnabled(msg.value);

        JST_CHECK(PersistSettings([&](Settings& settings) {
            settings.maps.highResolution = msg.value;
        }));

        return Result::SUCCESS;
    }

    Result handle(const MailSetMapsRendering& msg) {
        state.maps.rendering = msg.value;
        Render::Components::GeoMap::SetRenderingOptions(msg.value);
        return PersistSettings([&](Settings& settings) {
            settings.maps.rendering = msg.value;
        });
    }

    Result handle(const MailClearMapsCache&) {
        auto& maps = state.maps;
        if (maps.task.valid()) {
            return Result::SUCCESS;
        }

        maps.clearing = true;
        maps.task = std::async(std::launch::async, [] {
            CacheResult result;
            result.cleared = Tiles::ClearCache() == Result::SUCCESS;
            result.available = Tiles::CacheSize(result.bytes) == Result::SUCCESS;
            return result;
        });

        return Result::SUCCESS;
    }

    void reconcile() {
        auto& maps = state.maps;
        if (maps.task.valid()) {
            if (maps.task.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
                return;
            }
            const auto result = maps.task.get();
            maps.measured = true;
            maps.cacheAvailable = result.available;
            maps.cacheBytes = result.available ? std::optional<U64>(result.bytes) : std::nullopt;
            if (maps.clearing) {
                maps.clearing = false;
                if (result.cleared) {
                    callbacks.notify(Sakura::ToastType::Success, 3000, "Downloaded maps cleared.");
                } else {
                    callbacks.notify(Sakura::ToastType::Error, 5000, "Failed to clear downloaded maps.");
                }
            }
            return;
        }

        const bool shown = state.modal.content == ModalContent::Settings &&
                           state.settings.section == SettingsSection::Maps;
        if (!shown) {
            maps.measured = false;
            return;
        }
        if (maps.measured) {
            return;
        }

        maps.cacheBytes.reset();
        maps.task = std::async(std::launch::async, [] {
            CacheResult result;
            result.available = Tiles::CacheSize(result.bytes) == Result::SUCCESS;
            return result;
        });
    }

    void shutdown() {
        if (state.maps.task.valid()) {
            state.maps.task.wait();
        }
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_MAPS_HH
