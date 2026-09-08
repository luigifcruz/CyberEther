#ifndef JETSTREAM_DOMAINS_DSP_ADSB_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_ADSB_MODULE_IMPL_HH

#include <map>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <jetstream/domains/dsp/adsb/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/surface.hh>
#include <jetstream/tools/snapshot.hh>
#include <jetstream/render/base/texture.hh>
#include <jetstream/render/base/surface.hh>
#include <jetstream/render/components/geomap.hh>

#include "map_layers.hh"

namespace Jetstream::Modules {

struct AdsbImpl : public Module::Impl, public DynamicConfig<Adsb> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    struct AircraftInfo {
        U32 icao = 0;
        std::string callsign;
        I32 altitude = 0;
        F32 speed = 0.0f;
        F32 heading = 0.0f;
        F64 latitude = 0.0;
        F64 longitude = 0.0;
        bool hasCallsign = false;
        bool hasAltitude = false;
        bool hasVelocity = false;
        bool hasPosition = false;
        U64 lastPositionTimestamp = 0;  // Steady-clock milliseconds.

        // Position history for drawing tracks.
        std::vector<std::pair<F64, F64>> track;  // (lat, lon)

        // CPR tracking state.
        I32 rawLatEven = 0;
        I32 rawLonEven = 0;
        I32 rawLatOdd = 0;
        I32 rawLonOdd = 0;
        U64 evenTimestamp = 0;
        U64 oddTimestamp = 0;
    };

    JETSTREAM_API std::string getAircraftTable() const;
    void updateAircraftTable();

 protected:
    static constexpr U64 maxTrackPoints = AdsbMapState::MaxTrackPoints;

    Tensor input;
    Tools::Snapshot<std::string> aircraftTable{std::string("No aircraft detected.")};

    mutable std::mutex aircraftMutex;
    std::map<U32, AircraftInfo> aircraftMap;
    bool aircraftTableDirty = true;
    Extent2D<U64> viewSize = {512, 512};
    F32 viewScale = 1.0f;
    std::optional<Render::Components::GeoMap::Uniforms> savedMapView;
    AdsbMapAutoFit mapAutoFit;
    std::unique_ptr<Render::Components::GeoMap> geoMapComponent;
    std::shared_ptr<AdsbMapState> mapState;
    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Surface> renderSurface;

    Result createPresent();
    Result destroyPresent();
    Result present();

};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_ADSB_MODULE_IMPL_HH
