#ifndef JETSTREAM_DOMAINS_DSP_ADSB_MAP_LAYERS_HH
#define JETSTREAM_DOMAINS_DSP_ADSB_MAP_LAYERS_HH

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "jetstream/render/components/geomap.hh"

namespace Jetstream::Modules {

// Presentation-only snapshot. The module copies decoder state once under its
// mutex before presenting the map; all radar overlay passes see the same aircraft.
// Nothing in the map renderer knows this type or depends on the ADS-B decoder.
struct AdsbMapAircraft {
    U32 icao = 0;
    std::string callsign;
    F32 latitude = 0.0f;
    F32 longitude = 0.0f;
    F32 heading = 0.0f;
    F32 altitude = 0.0f;
    F32 groundSpeed = 0.0f;
    bool hasAltitude = false;
    bool hasVelocity = false;
    std::optional<F32> positionAgeSeconds;
    std::vector<std::pair<F64, F64>> track;  // latitude, longitude
};

struct AdsbMapState {
    static constexpr U64 MaxAircraft = 256;
    static constexpr U64 MaxTrackPoints = 128;
    std::vector<AdsbMapAircraft> aircraft;
};

// One-shot startup policy, retained by the module across surface recreation.
// The explicit steady-clock time keeps the collection window testable without sleeps.
class AdsbMapAutoFit {
 public:
    static constexpr U64 CollectionMilliseconds = 2000;
    JETSTREAM_API std::optional<Render::Components::MapContext::Uniforms> update(
        std::span<const AdsbMapAircraft> aircraft,
        const Render::Components::MapContext& context,
        std::span<const MouseEvent> mouseEvents, U64 now);

 private:
    bool pending = true;
    std::optional<U64> firstPositionTimestamp;
};

struct AdsbMapTarget {
    U64 index;
    Extent2D<F32> ndc;
    std::optional<F32> heading;  // Radians clockwise from screen north.
};

// CPU helpers shared by the module-owned radar overlay and regression tests.
// Uniform scaling for tracking visuals, independent of map/DPI scale.
inline constexpr F32 AdsbTrackingScale = 1.25f;
inline constexpr F32 AdsbStaleSeconds = 15.0f;
inline constexpr F32 AdsbVisibleSeconds = 120.0f;
JETSTREAM_API bool AdsbTargetVisible(const AdsbMapAircraft& aircraft);
JETSTREAM_API std::string FormatAdsbDataBlock(const AdsbMapAircraft& aircraft);
JETSTREAM_API std::vector<Extent2D<F32>> AdsbHistoryDots(
    const AdsbMapAircraft& aircraft, const Render::Components::MapContext& context);
JETSTREAM_API std::vector<AdsbMapTarget> AdsbVisibleTargets(
    std::span<const AdsbMapAircraft> aircraft, const Render::Components::MapContext& context);

struct AdsbLabelBox {
    F32 x, y, width, height;  // Logical pixels, origin top left.
};
struct AdsbLabelPlacement {
    AdsbLabelBox box;
    U8 direction;
};
JETSTREAM_API AdsbLabelPlacement PlaceAdsbDataBlock(
    const Extent2D<F32>& anchor, const Extent2D<F32>& size,
    const Extent2D<F32>& viewport, std::span<const AdsbLabelBox> occupied,
    U8 preferredDirection = 0);

// The module contributes its tracking overlay to an otherwise unchanged base map.
// The same public GeoMap/MapLayer API is available to out-of-tree plugins.
Result AddAdsbMapLayers(Render::Components::GeoMap& map,
                       const std::shared_ptr<const AdsbMapState>& state);

}  // namespace Jetstream::Modules

#endif
