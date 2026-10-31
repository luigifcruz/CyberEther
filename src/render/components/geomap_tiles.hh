#ifndef JETSTREAM_RENDER_COMPONENTS_GEOMAP_TILES_HH
#define JETSTREAM_RENDER_COMPONENTS_GEOMAP_TILES_HH

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "jetstream/render/components/map_context.hh"
#include "jetstream/render/components/map_options.hh"

namespace Jetstream::Render::Components {

class GeoMapTiles {
 public:
    enum class Line : U8 {
        Stream,
        Canal,
        River,
        Building,
        MinorRoad,
        SecondaryRoad,
        PrimaryRoad,
        Motorway,
        Rail,
        StateBorder,
        CountryBorder,
        Coastline,
        Taxiway,
        Runway,
    };
    static constexpr U64 LineCount = 14;

    enum class Fill : U8 {
        Land,
        Urban,
        Water,
        Airport,
        Building,
        BuildingMid,
        BuildingTall,
    };
    static constexpr U64 FillCount = 7;
    static constexpr U64 FirstBuildingFill = static_cast<U64>(Fill::Building);

    enum class Place : U8 {
        City,
        Town,
        Village,
        Suburb,
        Hamlet,
        Airport,
        Airfield,
    };

    struct Key {
        U8 z = 0;
        U32 x = 0;
        U32 y = 0;

        U64 id() const;
        bool operator==(const Key&) const = default;
    };

    class Source {
     public:
        enum class Status {
            Found,
            Missing,
            Failed,
        };

        virtual ~Source() = default;
        virtual std::string name() const = 0;
        virtual std::string attribution() const = 0;
        virtual U8 maxZoom() const = 0;
        virtual Status fetch(const Key& key, std::string& data,
                             const std::function<bool()>& cancelled) = 0;
    };

    class OpenFreeMap final : public Source {
     public:
        OpenFreeMap();
        ~OpenFreeMap() override;

        std::string name() const override;
        std::string attribution() const override;
        U8 maxZoom() const override;
        Status fetch(const Key& key, std::string& data,
                     const std::function<bool()>& cancelled) override;

     private:
        struct Impl;
        std::unique_ptr<Impl> pimpl;
    };

    struct Mesh {
        std::vector<F32> positions;
        std::vector<U32> indices;
    };

    struct Label {
        F32 lon = 0.0f;
        F32 lat = 0.0f;
        Place kind = Place::Village;
        std::string name;
    };

    struct Geometry {
        std::array<std::vector<F32>, LineCount> lines;
        std::array<std::vector<F32>, LineCount> lineFades;
        std::array<Mesh, FillCount> fills;
        std::array<std::vector<F32>, FillCount> fillFades;
        std::array<U64, FillCount> heldFillIndices{};
        std::vector<Label> labels;
        F64 originLon = 0.0;
        F64 originLat = 0.0;
        U64 revision = 0;
    };

    struct Config {
        MapRenderingOptions rendering;
        F32 fadeStart = 7.0f;
        F32 fadeEnd = 8.0f;
        F32 tilePixels = 512.0f;
        F32 fadeSeconds = 0.6f;
        U64 maxTiles = 64;
        U64 workers = 4;
        U64 memoryBudgetBytes = 160ull << 20;
        U64 diskBudgetBytes = 1024ull << 20;
        std::array<U64, LineCount> lineCapacity{};
        U64 fillVertexCapacity = 0;
        U64 fillIndexCapacity = 0;
        U64 buildingVertexCapacity = 0;
        U64 buildingIndexCapacity = 0;
    };

    explicit GeoMapTiles(std::unique_ptr<Source> source);
    ~GeoMapTiles();

    GeoMapTiles(const GeoMapTiles&) = delete;
    GeoMapTiles& operator=(const GeoMapTiles&) = delete;

    void start(const Config& config);
    void stop();

    bool update(const MapContext& context);

    F32 opacity() const;
    F32 zoom() const;
    F32 time() const;
    const Geometry& geometry() const;
    std::string attribution() const;

    static F32 DetailZoom(const MapContext& context, F32 tilePixels);

    static void SetEnabled(bool enabled);
    static bool Enabled();
    static Result CacheSize(U64& bytes);
    static Result ClearCache();

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Render::Components

#endif  // JETSTREAM_RENDER_COMPONENTS_GEOMAP_TILES_HH
