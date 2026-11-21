#include "maps.hh"

namespace Jetstream {

MapsActions::MapsActions(DefaultCompositorState& state,
                         DefaultCompositorCallbacks& callbacks) :
    state(state),
    callbacks(callbacks) {}

Result MapsActions::handle(const MailSetMapsHighResolution& msg) {
    state.maps.highResolution = msg.value;
    Tiles::SetEnabled(msg.value);

    JST_CHECK(PersistSettings([&](Settings& settings) {
        settings.maps.highResolution = msg.value;
    }));

    return Result::SUCCESS;
}

Result MapsActions::handle(const MailSetMapsRendering& msg) {
    state.maps.rendering = msg.value;
    Render::Components::GeoMap::SetRenderingOptions(msg.value);
    return PersistSettings([&](Settings& settings) {
        settings.maps.rendering = msg.value;
    });
}

Result MapsActions::handle(const MailClearMapsCache&) {
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

void MapsActions::reconcile() {
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

    const bool shown = state.interface.settingsTab && state.settings.section == SettingsSection::Maps;
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

void MapsActions::shutdown() {
    if (state.maps.task.valid()) {
        state.maps.task.wait();
    }
}

}  // namespace Jetstream
