#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_MAPS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_MAPS_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/settings/maps.hh"

namespace Jetstream {

struct MapsSettingsPresenter {
    const PresenterContext& context;

    explicit MapsSettingsPresenter(const PresenterContext& context) : context(context) {}

    MapsSettingsView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;
        const auto& maps = context.state.maps;

        return MapsSettingsView::Config{
            .highResolution = maps.highResolution,
            .rendering = maps.rendering,
            .cacheAvailable = maps.cacheAvailable,
            .cacheBytes = maps.cacheBytes,
            .measuring = !maps.clearing && (!maps.measured || maps.task.valid()),
            .clearing = maps.clearing,
            .onHighResolutionChange = [enqueue](bool value) {
                enqueue(MailSetMapsHighResolution{.value = value});
            },
            .onClearCache = [enqueue]() {
                enqueue(MailClearMapsCache{});
            },
            .onRenderingChange = [enqueue](const Render::Components::MapRenderingOptions& value) {
                enqueue(MailSetMapsRendering{.value = value});
            },
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_SETTINGS_MAPS_HH
