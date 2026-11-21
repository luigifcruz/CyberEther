#include "maps.hh"

namespace Jetstream {

MapsSettingsPresenter::MapsSettingsPresenter(const PresenterContext& context) : context(context) {}

MapsSettingsView::Config MapsSettingsPresenter::build() const {
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

}  // namespace Jetstream
