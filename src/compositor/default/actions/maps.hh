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

namespace Jetstream {

struct MapsActions {
    using CacheResult = DefaultCompositorState::MapsState::CacheResult;
    using Tiles = Render::Components::GeoMapTiles;

    DefaultCompositorState& state;
    DefaultCompositorCallbacks& callbacks;

    MapsActions(DefaultCompositorState& state,
                DefaultCompositorCallbacks& callbacks);
    Result handle(const MailSetMapsHighResolution& msg);
    Result handle(const MailSetMapsRendering& msg);
    Result handle(const MailClearMapsCache&);
    void reconcile();
    void shutdown();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_MAPS_HH
