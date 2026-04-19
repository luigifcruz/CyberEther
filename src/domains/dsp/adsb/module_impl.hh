#ifndef JETSTREAM_DOMAINS_DSP_ADSB_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_DSP_ADSB_MODULE_IMPL_HH

#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <jetstream/domains/dsp/adsb/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/surface.hh>
#include <jetstream/tools/snapshot.hh>
#include <jetstream/render/base/buffer.hh>
#include <jetstream/render/base/texture.hh>
#include <jetstream/render/base/surface.hh>
#include <jetstream/render/base/program.hh>
#include <jetstream/render/base/vertex.hh>
#include <jetstream/render/base/draw.hh>
#include <jetstream/render/components/geomap.hh>
#include <jetstream/render/components/text.hh>

namespace Jetstream::Modules {

struct AdsbImpl : public Module::Impl, public DynamicConfig<Adsb> {
 public:
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

    std::string getAircraftTable() const;
    void updateOutputTensors();

 protected:
    static constexpr U64 maxAircraft = 256;
    static constexpr U64 maxTrackPoints = 128;

    Tensor input;
    Tensor aircraft;
    Tensor aircraftCount;
    Tools::Snapshot<std::string> aircraftTable{std::string("No aircraft detected.")};
    std::vector<F32> aircraftInstances;

    mutable std::mutex aircraftMutex;
    std::map<U32, AircraftInfo> aircraftMap;
    bool aircraftTableDirty = true;
    std::string hoveredFlightId;
    Extent2D<F32> hoveredFlightNdc = {0.0f, 0.0f};
    bool hasLastCursorSample = false;
    Extent2D<F32> lastCursorSample = {0.0f, 0.0f};

    // Generic interaction state processed by ProcessSurfaceInteraction.
    SurfaceInteractionState surfaceInteraction;

    // Map interaction state.
    struct MapInteraction {
        F32 centerLon = 0.0f;
        F32 centerLat = 0.0f;
        F32 zoom = 1.0f;  // Mercator zoom level (0-18).
        F32 scale = 1.0f;
        Extent2D<U64> viewSize = {512, 512};
        bool dragging = false;
        F32 dragAnchorX = 0.0f;
        F32 dragAnchorY = 0.0f;
        F32 dragStartLon = 0.0f;
        F32 dragStartLat = 0.0f;
        bool viewChanged = false;
    } mapInteraction;

    // Aircraft rendering uniforms.
    struct {
        float centerLon;
        float centerLat;
        float zoom;
        float aspectRatio;
        float surfaceScale;
        int viewWidth;
        int viewHeight;
        int aircraftCount;
        int _pad0;
    } aircraftUniforms;

    // GeoMap render component.
    std::unique_ptr<Render::Components::GeoMap> geoMapComponent;
    std::shared_ptr<Render::Components::Text> hoverText;

    // Aircraft rendering members.
    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;

    std::shared_ptr<Render::Buffer> aircraftBuffer;
    std::shared_ptr<Render::Buffer> aircraftUniformBuffer;

    std::shared_ptr<Render::Texture> framebufferTexture;

    std::shared_ptr<Render::Program> aircraftProgram;

    std::shared_ptr<Render::Surface> renderSurface;

    std::shared_ptr<Render::Vertex> aircraftVertex;

    std::shared_ptr<Render::Draw> drawAircraftDots;

    // Track rendering members.
    struct {
        float centerLon;
        float centerLat;
        float zoom;
        float aspectRatio;
        float surfaceScale;
        float lineWidth;
        float colorR;
        float colorG;
        float colorB;
        float viewportWidth;
        float viewportHeight;
    } trackGpuUniforms;

    std::vector<F32> trackSegments;

    std::shared_ptr<Render::Buffer> trackQuadBuffer;
    std::shared_ptr<Render::Buffer> trackInstanceBuffer;
    std::shared_ptr<Render::Buffer> trackUniformBuffer;
    std::shared_ptr<Render::Vertex> trackVertex;
    std::shared_ptr<Render::Draw> trackDraw;
    std::shared_ptr<Render::Program> trackProgram;

    Result createPresent();
    Result destroyPresent();
    Result present();

    void processMapInteraction(std::vector<SurfaceEvent>&& surfaceEvents,
                               std::vector<MouseEvent>&& mouseEvents);
    void clampMapView();
    void updateHoveredFlightFromCursor(const Extent2D<F32>& cursorPosNormalized);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_ADSB_MODULE_IMPL_HH
