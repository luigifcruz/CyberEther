#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <thread>

#include "jetstream/platform.hh"
#include "render/components/geomap_tiles.hh"

using namespace Jetstream;
using namespace Jetstream::Render::Components;
using namespace std::chrono_literals;

namespace {

std::string Varint(U64 value) {
    std::string result;
    do {
        const U8 byte = value & 0x7f;
        value >>= 7;
        result.push_back(static_cast<char>(byte | (value ? 0x80 : 0)));
    } while (value);
    return result;
}

std::string Bytes(U64 field, const std::string& value) {
    return Varint((field << 3) | 2) + Varint(value.size()) + value;
}

enum class Encoding { Packed, SplitPacked, Unpacked, Mixed };

std::string Repeated(U32 field, const std::vector<U64>& values, Encoding encoding) {
    std::string result;
    if (encoding == Encoding::Packed || encoding == Encoding::SplitPacked) {
        std::string packed;
        for (U64 i = 0; i < values.size(); ++i) {
            packed += Varint(values[i]);
            if (encoding == Encoding::SplitPacked && i % 3 == 2) {
                result += Bytes(field, packed);
                packed.clear();
            }
        }
        return result + Bytes(field, packed);
    }
    for (U64 i = 0; i < values.size(); ++i) {
        result += encoding == Encoding::Mixed && i % 2
                      ? Bytes(field, Varint(values[i]))
                      : Varint(field << 3) + Varint(values[i]);
    }
    return result;
}

std::string GeometryTileFields(U32 type, const std::string& geometry, const std::string& tags) {
    const auto feature = tags + Varint(24) + Varint(type) + geometry;
    const auto layer = Bytes(1, type == 1 ? "aerodrome_label" : type == 2 ? "aeroway" : "water") +
                       Bytes(2, feature) + Bytes(3, "class") + Bytes(3, "iata") +
                       Bytes(4, Bytes(1, type == 1 ? "international" : type == 2 ? "runway" : "ocean")) +
                       Bytes(4, Bytes(1, "TEST")) +
                       Varint(40) + Varint(4096) + Varint(120) + Varint(2);
    return Bytes(3, layer);
}

std::string GeometryTile(U32 type, const std::vector<U64>& commands,
                         Encoding geometry = Encoding::Packed, Encoding tags = Encoding::Packed) {
    return GeometryTileFields(type, Repeated(4, commands, geometry), Repeated(2, {0, 0, 1, 1}, tags));
}

std::string WaterTile() {
    // A clockwise polygon covering the tile's 4096 x 4096 extent.
    return GeometryTile(3, {9, 0, 0, 26, 8192, 0, 0, 8192, 8191, 0, 15});
}

std::string LayerTile(const std::string& name, U32 type,
                      const std::vector<U64>& commands, const std::string& kind) {
    const auto feature = Repeated(2, {0, 0, 1, 1, 2, 2, 3, 1}, Encoding::Packed) +
                         Varint(24) + Varint(type) + Repeated(4, commands, Encoding::Packed);
    const auto layer = Bytes(1, name) + Bytes(2, feature) +
                       Bytes(3, "class") + Bytes(3, "name") + Bytes(3, "admin_level") + Bytes(3, "iata") +
                       Bytes(4, Bytes(1, kind)) + Bytes(4, Bytes(1, "TEST")) +
                       Bytes(4, Varint(40) + Varint(2)) +
                       Varint(40) + Varint(4096) + Varint(120) + Varint(2);
    return Bytes(3, layer);
}

using BuildingRing = std::vector<std::array<I32, 2>>;

std::string BuildingTile(const std::vector<BuildingRing>& rings) {
    std::vector<U64> commands;
    I32 x = 0;
    I32 y = 0;
    auto point = [&](const auto& p) {
        auto zigzag = [](I64 value) -> U64 { return value < 0 ? -value * 2 - 1 : value * 2; };
        commands.push_back(zigzag(p[0] - x));
        commands.push_back(zigzag(p[1] - y));
        x = p[0];
        y = p[1];
    };
    for (const auto& ring : rings) {
        commands.push_back(9);
        point(ring.front());
        commands.push_back(((ring.size() - 1) << 3) | 2);
        for (U64 i = 1; i < ring.size(); ++i) point(ring[i]);
        commands.push_back(15);
    }
    const auto feature = Varint(24) + Varint(3) + Repeated(4, commands, Encoding::Packed);
    const auto layer = Bytes(1, "building") + Bytes(2, feature) +
                       Varint(40) + Varint(4096) + Varint(120) + Varint(2);
    return Bytes(3, layer);
}

std::array<F64, 2> TilePoint(F64 lon, F64 lat) {
    constexpr F64 pi = 3.14159265358979323846;
    return {(lon + 180.0) / 360.0,
            (1.0 - std::asinh(std::tan(lat * pi / 180.0)) / pi) * 0.5};
}

F64 TileArea(const GeoMapTiles::Geometry& geometry, GeoMapTiles::Fill fill) {
    const auto& mesh = geometry.fills[static_cast<U64>(fill)];
    F64 area = 0.0;
    REQUIRE(mesh.indices.size() % 3 == 0);
    for (U64 i = 0; i < mesh.indices.size(); i += 3) {
        std::array<std::array<F64, 2>, 3> p;
        for (U64 j = 0; j < 3; ++j) {
            const U64 index = mesh.indices[i + j] * 2;
            REQUIRE(index + 1 < mesh.positions.size());
            p[j] = TilePoint(mesh.positions[index] + geometry.originLon,
                             mesh.positions[index + 1] + geometry.originLat);
        }
        area += std::abs((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) -
                         (p[2][0] - p[0][0]) * (p[1][1] - p[0][1])) * 0.5;
    }
    return area;
}

struct GeometryCase {
    const char* name;
    U32 type;
    std::vector<U64> commands;
};

const std::vector<GeometryCase> MalformedGeometries = {
    {"polygon empty", 3, {}},
    {"polygon truncated MoveTo", 3, {9, 0}},
    {"polygon MoveTo only", 3, {9, 0, 0}},
    {"polygon ClosePath only", 3, {15}},
    {"polygon missing ClosePath", 3, {9, 0, 0, 18, 8192, 0, 0, 8192}},
    {"polygon too few edges", 3, {9, 0, 0, 10, 8192, 0, 15}},
    {"polygon invalid MoveTo count", 3, {17, 0, 0, 0, 0, 18, 8192, 0, 0, 8192, 15}},
    {"polygon invalid ClosePath count", 3, {9, 0, 0, 18, 8192, 0, 0, 8192, 23}},
    {"polygon zero-count LineTo", 3, {9, 0, 0, 2, 15}},
    {"polygon line after closure", 3, {9, 0, 0, 18, 8192, 0, 0, 8192, 15, 10, 1, 0}},
    {"polygon next ring before closure", 3,
        {9, 0, 0, 18, 8192, 0, 0, 8192, 9, 0, 0, 18, 2, 0, 0, 2, 15}},
    {"polygon zero-length ring", 3, {9, 0, 0, 18, 0, 0, 0, 0, 15}},
    {"polygon zero-length edge", 3, {9, 0, 0, 26, 0, 0, 8192, 0, 0, 8192, 15}},
    {"polygon repeated closing vertex", 3, {9, 0, 0, 26, 8192, 0, 0, 8192, 8191, 8191, 15}},
    {"polygon zero area", 3, {9, 0, 0, 18, 2, 2, 2, 2, 15}},
    {"polygon orphan interior ring", 3, {9, 0, 0, 26, 0, 8192, 8192, 0, 0, 8191, 15}},
    {"polygon oversized command", 3, {9 + (1ull << 32), 0, 0, 18, 8192, 0, 0, 8192, 15}},
    {"polygon oversized coordinate", 3, {9, 1ull << 32, 0, 18, 8192, 0, 0, 8192, 15}},
    {"line MoveTo only", 2, {9, 0, 0}},
    {"line zero-length edge", 2, {9, 0, 0, 10, 0, 0}},
    {"line invalid MoveTo count", 2, {17, 0, 0, 2, 2, 10, 2, 2}},
    {"line ClosePath", 2, {9, 0, 0, 10, 2, 2, 15}},
    {"point empty", 1, {}},
    {"point LineTo", 1, {9, 0, 0, 10, 2, 2}},
    {"point repeated MoveTo", 1, {9, 0, 0, 9, 2, 2}},
    {"point zero-count MoveTo", 1, {1}},
};

std::vector<std::pair<std::string, std::string>> MalformedTiles() {
    std::vector<std::pair<std::string, std::string>> result;
    for (const auto& input : MalformedGeometries) {
        result.emplace_back(input.name, GeometryTile(input.type, input.commands));
    }
    const auto tags = Repeated(2, {0, 0, 1, 1}, Encoding::Packed);
    result.emplace_back("varint split across packed fields", GeometryTileFields(3,
        Bytes(4, std::string("\x89", 1)) +
        Bytes(4, std::string("\x00\x00\x00\x12\x02\x00\x00\x02\x0f", 9)), tags));
    result.emplace_back("overflowing varint", GeometryTileFields(3,
        Bytes(4, std::string(9, '\x80') + '\x02'), tags));
    result.emplace_back("building incomplete ring", WaterTile() + BuildingTile({{{1024, 1024}}}));
    result.emplace_back("building orphan courtyard", WaterTile() + BuildingTile({
        {{1024, 1024}, {1024, 3072}, {3072, 3072}, {3072, 1024}},
    }));
    return result;
}

struct TileCache {
    TileCache() {
        static std::atomic<U64> sequence{0};
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() /
               ("cyberether-tile-test-" + std::to_string(nonce) + "-" +
                std::to_string(sequence.fetch_add(1)));
    }

    ~TileCache() {
        std::error_code error;
        std::filesystem::remove_all(root, error);
    }

    std::filesystem::path tilePath() const { return root / "0" / "0" / "0.mvt"; }
    std::filesystem::path root;
};

class TileSource final : public GeoMapTiles::Source {
 public:
    TileSource(const TileCache& cache, std::string data)
        : directory(Platform::PathToUtf8(cache.root)), data(std::move(data)) {}

    // An absolute source directory keeps test downloads out of the user's cache.
    std::string name() const override { return directory; }
    std::string attribution() const override { return "Test tiles"; }
    U8 maxZoom() const override { return 0; }

    Status fetch(const GeoMapTiles::Key&, std::string& result,
                 const std::function<bool()>&) override {
        ++fetches;
        result = data;
        return Status::Found;
    }

    std::atomic<U64> fetches{0};

 private:
    std::string directory;
    std::string data;
};

GeoMapTiles::Config TileConfig() {
    GeoMapTiles::Config config;
    config.workers = 1;
    config.fadeSeconds = 0.0f;
    config.fillVertexCapacity = 1u << 20;
    config.fillIndexCapacity = 3u << 20;
    config.buildingVertexCapacity = 1u << 20;
    config.buildingIndexCapacity = 3u << 20;
    config.lineCapacity.fill(1024);
    return config;
}

MapContext TileContext() {
    MapContext context;
    auto view = context.view;
    view.centerLon = -122.4f;
    view.centerLat = 37.8f;
    view.zoom = 8.0f;
    view.detailZoom = 8.0f;
    context.update(view);
    return context;
}

void Pump(GeoMapTiles& tiles, const MapContext& context,
          std::chrono::milliseconds duration) {
    const auto end = std::chrono::steady_clock::now() + duration;
    do {
        tiles.update(context);
        std::this_thread::sleep_for(5ms);
    } while (std::chrono::steady_clock::now() < end);
}

template <typename Predicate>
bool WaitFor(GeoMapTiles& tiles, const MapContext& context, Predicate&& ready) {
    const auto end = std::chrono::steady_clock::now() + 2s;
    do {
        tiles.update(context);
        if (ready(tiles.geometry())) return true;
        std::this_thread::sleep_for(5ms);
    } while (std::chrono::steady_clock::now() < end);
    return false;
}

bool WaitForWater(GeoMapTiles& tiles, const MapContext& context) {
    return WaitFor(tiles, context, [](const auto& geometry) {
        return !geometry.fills[static_cast<U64>(GeoMapTiles::Fill::Water)].indices.empty();
    });
}

}  // namespace

TEST_CASE("GeoMap restores cached detail after restarting at the same view", "[render][geomap][tiles]") {
    TileCache cache;
    auto source = std::make_unique<TileSource>(cache, WaterTile());
    const auto* observed = source.get();
    GeoMapTiles tiles(std::move(source));
    const auto context = TileContext();
    const auto config = TileConfig();
    tiles.start(config);
    REQUIRE(WaitForWater(tiles, context));
    // Drain the completion-triggered assembly before restarting, so a pending
    // download cannot mask the missing restart-triggered assembly.
    Pump(tiles, context, 350ms);
    const auto count = tiles.geometry().fills[static_cast<U64>(GeoMapTiles::Fill::Water)].indices.size();
    REQUIRE(observed->fetches.load() == 1);

    tiles.stop();
    REQUIRE(tiles.geometry().fills[static_cast<U64>(GeoMapTiles::Fill::Water)].indices.empty());
    tiles.start(config);
    REQUIRE(WaitForWater(tiles, context));
    CHECK(tiles.geometry().fills[static_cast<U64>(GeoMapTiles::Fill::Water)].indices.size() == count);
    CHECK(observed->fetches.load() == 1);
}

TEST_CASE("GeoMap rejects malformed downloads without poisoning the cache", "[render][geomap][tiles]") {
    for (const auto& [name, payload] : MalformedTiles()) {
        DYNAMIC_SECTION(name) {
            TileCache cache;
            const auto context = TileContext();
            {
                auto source = std::make_unique<TileSource>(cache, payload);
                const auto* observed = source.get();
                GeoMapTiles tiles(std::move(source));
                tiles.start(TileConfig());
                Pump(tiles, context, 350ms);
                CHECK(tiles.geometry().fills[static_cast<U64>(GeoMapTiles::Fill::Land)].indices.empty());
                tiles.stop();  // Join the worker before inspecting its disk writes.
                REQUIRE(observed->fetches.load() == 1);
                REQUIRE_FALSE(std::filesystem::exists(cache.tilePath()));
            }

            auto source = std::make_unique<TileSource>(cache, WaterTile());
            const auto* observed = source.get();
            GeoMapTiles tiles(std::move(source));
            tiles.start(TileConfig());
            REQUIRE(WaitForWater(tiles, context));
            CHECK(observed->fetches.load() == 1);
        }
    }
}

TEST_CASE("GeoMap replaces malformed cached geometry with a valid download", "[render][geomap][tiles]") {
    for (const auto& [name, payload] : MalformedTiles()) {
        DYNAMIC_SECTION(name) {
            TileCache cache;
            std::filesystem::create_directories(cache.tilePath().parent_path());
            {
                std::ofstream file(cache.tilePath(), std::ios::binary);
                file << payload;
                REQUIRE(file.good());
            }

            auto source = std::make_unique<TileSource>(cache, WaterTile());
            const auto* observed = source.get();
            GeoMapTiles tiles(std::move(source));
            tiles.start(TileConfig());
            REQUIRE(WaitForWater(tiles, TileContext()));
            CHECK(observed->fetches.load() == 1);
        }
    }
}

TEST_CASE("GeoMap accepts equivalent packed and unpacked repeated fields", "[render][geomap][tiles]") {
    const std::vector<GeometryCase> inputs = {
        {"multipoint", 1, {17, 10, 14, 3, 9}},
        {"multiline", 2, {9, 4, 4, 18, 0, 16, 16, 0, 9, 17, 17, 10, 4, 8}},
        {"water", 3, {9, 0, 0, 26, 8192, 0, 0, 8192, 8191, 0, 15}},
    };
    auto load = [](const std::string& payload) -> std::optional<GeoMapTiles::Geometry> {
        TileCache cache;
        GeoMapTiles tiles(std::make_unique<TileSource>(cache, payload));
        tiles.start(TileConfig());
        if (!WaitFor(tiles, TileContext(), [](const auto& geometry) {
            return !geometry.fills[static_cast<U64>(GeoMapTiles::Fill::Land)].indices.empty();
        })) return std::nullopt;
        return tiles.geometry();
    };
    for (const auto& input : inputs) {
        CAPTURE(input.name);
        const auto canonical = load(GeometryTile(input.type, input.commands));
        REQUIRE(canonical.has_value());
        for (const auto geometryEncoding : {Encoding::Packed, Encoding::SplitPacked,
                                            Encoding::Unpacked, Encoding::Mixed}) {
            for (const auto tagEncoding : {Encoding::Packed, Encoding::SplitPacked,
                                           Encoding::Unpacked, Encoding::Mixed}) {
                CAPTURE(static_cast<int>(geometryEncoding), static_cast<int>(tagEncoding));
                const auto decoded = load(GeometryTile(input.type, input.commands, geometryEncoding, tagEncoding));
                REQUIRE(decoded.has_value());
                CHECK(decoded->lines == canonical->lines);
                for (U64 f = 0; f < GeoMapTiles::FillCount; ++f) {
                    CHECK(decoded->fills[f].positions == canonical->fills[f].positions);
                    CHECK(decoded->fills[f].indices == canonical->fills[f].indices);
                }
                REQUIRE(decoded->labels.size() == canonical->labels.size());
                for (U64 i = 0; i < decoded->labels.size(); ++i) {
                    CHECK(decoded->labels[i].name == canonical->labels[i].name);
                    CHECK(decoded->labels[i].kind == canonical->labels[i].kind);
                    CHECK(decoded->labels[i].lon == canonical->labels[i].lon);
                    CHECK(decoded->labels[i].lat == canonical->labels[i].lat);
                }
            }
        }
    }
}

TEST_CASE("GeoMap caches valid polygons with no coverage inside the tile", "[render][geomap][tiles]") {
    TileCache cache;
    // A clockwise square wholly outside the tile extent, with valid edges.
    const auto payload = GeometryTile(3, {9, 10000, 10000, 26, 2000, 0, 0, 2000, 1999, 0, 15});
    GeoMapTiles tiles(std::make_unique<TileSource>(cache, payload));
    tiles.start(TileConfig());
    REQUIRE(WaitFor(tiles, TileContext(), [](const auto& geometry) {
        return !geometry.fills[static_cast<U64>(GeoMapTiles::Fill::Land)].indices.empty();
    }));
    CHECK(tiles.geometry().fills[static_cast<U64>(GeoMapTiles::Fill::Water)].indices.empty());
    CHECK(std::filesystem::exists(cache.tilePath()));
}

TEST_CASE("GeoMap accepts complete single and multipart geometries", "[render][geomap][tiles]") {
    // Command sequences from the vector-tile 2.1 specification, including
    // two polygons and a hole whose MoveTo remains relative to the last vertex.
    const std::vector<GeometryCase> inputs = {
        {"point", 1, {9, 50, 34}},
        {"multipoint", 1, {17, 10, 14, 3, 9}},
        {"line", 2, {9, 4, 4, 18, 0, 16, 16, 0}},
        {"multiline", 2, {9, 4, 4, 18, 0, 16, 16, 0, 9, 17, 17, 10, 4, 8}},
        {"polygon", 3, {9, 6, 12, 18, 10, 12, 24, 44, 15}},
        {"multipolygon with hole", 3,
            {9, 0, 0, 26, 20, 0, 0, 20, 19, 0, 15,
             9, 22, 2, 26, 18, 0, 0, 18, 17, 0, 15,
             9, 4, 13, 26, 0, 8, 8, 0, 0, 7, 15}},
    };
    for (const auto& input : inputs) {
        DYNAMIC_SECTION(input.name) {
            TileCache cache;
            auto source = std::make_unique<TileSource>(cache, GeometryTile(input.type, input.commands));
            GeoMapTiles tiles(std::move(source));
            tiles.start(TileConfig());
            REQUIRE(WaitFor(tiles, TileContext(), [&](const auto& geometry) {
                if (input.type == 1) return !geometry.labels.empty();
                if (input.type == 2) {
                    return !geometry.lines[static_cast<U64>(GeoMapTiles::Line::Runway)].empty();
                }
                return !geometry.fills[static_cast<U64>(GeoMapTiles::Fill::Water)].indices.empty();
            }));
            CHECK(std::filesystem::exists(cache.tilePath()));
        }
    }
}

TEST_CASE("GeoMap building footprints preserve courtyards and closed outlines", "[render][geomap][tiles]") {
    TileCache cache;
    const auto payload = BuildingTile({
        {{1024, 1024}, {3072, 1024}, {3072, 3072}, {1024, 3072}},
        {{1536, 1536}, {1536, 2560}, {2560, 2560}, {2560, 1536}},
    });
    GeoMapTiles tiles(std::make_unique<TileSource>(cache, payload));
    tiles.start(TileConfig());
    REQUIRE(WaitFor(tiles, TileContext(), [](const auto& geometry) {
        return !geometry.fills[static_cast<U64>(GeoMapTiles::Fill::Building)].indices.empty();
    }));
    const auto& geometry = tiles.geometry();
    CHECK(std::abs(TileArea(geometry, GeoMapTiles::Fill::Building) - 0.1875) < 1e-6);
    const auto& lines = geometry.lines[static_cast<U64>(GeoMapTiles::Line::Building)];
    REQUIRE(lines.size() == 8 * 4);
    for (U64 ring = 0; ring < 2; ++ring) {
        for (U64 edge = 0; edge < 4; ++edge) {
            const U64 at = ring * 16 + edge * 4;
            const U64 next = ring * 16 + ((edge + 1) % 4) * 4;
            CHECK(lines[at + 2] == lines[next]);
            CHECK(lines[at + 3] == lines[next + 1]);
        }
    }
}

TEST_CASE("GeoMap building outlines draw shared walls once", "[render][geomap][tiles]") {
    TileCache cache;
    const auto payload = BuildingTile({
        {{1024, 1024}, {3072, 1024}, {3072, 3072}, {1024, 3072}},
        {{3072, 1024}, {4000, 1024}, {4000, 3072}, {3072, 3072}},
    });
    GeoMapTiles tiles(std::make_unique<TileSource>(cache, payload));
    tiles.start(TileConfig());
    REQUIRE(WaitFor(tiles, TileContext(), [](const auto& geometry) {
        return !geometry.fills[static_cast<U64>(GeoMapTiles::Fill::Building)].indices.empty();
    }));
    CHECK(tiles.geometry().lines[static_cast<U64>(GeoMapTiles::Line::Building)].size() == 7 * 4);
}

TEST_CASE("GeoMap building budget overflow keeps the rest of the tile", "[render][geomap][tiles]") {
    TileCache cache;
    const auto payload = WaterTile() + BuildingTile({
        {{1024, 1024}, {3072, 1024}, {3072, 3072}, {1024, 3072}},
    });
    GeoMapTiles tiles(std::make_unique<TileSource>(cache, payload));
    auto config = TileConfig();
    config.buildingVertexCapacity = 1;
    tiles.start(config);
    REQUIRE(WaitForWater(tiles, TileContext()));
    const auto& geometry = tiles.geometry();
    for (U64 f = GeoMapTiles::FirstBuildingFill; f < GeoMapTiles::FillCount; ++f) {
        CHECK(geometry.fills[f].indices.empty());
    }
    CHECK(geometry.lines[static_cast<U64>(GeoMapTiles::Line::Building)].empty());
    CHECK_FALSE(geometry.fills[static_cast<U64>(GeoMapTiles::Fill::Land)].indices.empty());
}

TEST_CASE("GeoMap building clipping does not draw artificial tile boundaries", "[render][geomap][tiles]") {
    struct Case {
        const char* name;
        BuildingRing ring;
        F64 area;
        U64 edges;
    };
    const std::array<Case, 4> cases = {{
        {"buffered footprint", {{3072, 1024}, {5120, 1024}, {5120, 3072}, {3072, 3072}}, 0.125, 3},
        {"clipped footprint", {{3072, 1024}, {4096, 1024}, {4096, 3072}, {3072, 3072}}, 0.125, 3},
        {"surrounding footprint", {{-1024, -1024}, {5120, -1024}, {5120, 5120}, {-1024, 5120}}, 1.0, 0},
        {"outside footprint", {{5000, 5000}, {6000, 5000}, {6000, 6000}, {5000, 6000}}, 0.0, 0},
    }};
    for (const auto& input : cases) {
        DYNAMIC_SECTION(input.name) {
            TileCache cache;
            GeoMapTiles tiles(std::make_unique<TileSource>(cache, BuildingTile({input.ring})));
            tiles.start(TileConfig());
            REQUIRE(WaitFor(tiles, TileContext(), [](const auto& geometry) {
                return !geometry.fills[static_cast<U64>(GeoMapTiles::Fill::Land)].indices.empty();
            }));
            const auto& geometry = tiles.geometry();
            CHECK(std::abs(TileArea(geometry, GeoMapTiles::Fill::Building) - input.area) < 1e-6);
            const auto& lines = geometry.lines[static_cast<U64>(GeoMapTiles::Line::Building)];
            REQUIRE(lines.size() == input.edges * 4);
            for (U64 i = 0; i < lines.size(); i += 4) {
                const auto a = TilePoint(lines[i] + geometry.originLon, lines[i + 1] + geometry.originLat);
                const auto b = TilePoint(lines[i + 2] + geometry.originLon, lines[i + 3] + geometry.originLat);
                CHECK_FALSE((std::abs(a[0] - 1.0) < 1e-6 && std::abs(b[0] - 1.0) < 1e-6));
                for (const auto& p : {a, b}) {
                    CHECK(p[0] >= -1e-6);
                    CHECK(p[0] <= 1.0 + 1e-6);
                    CHECK(p[1] >= -1e-6);
                    CHECK(p[1] <= 1.0 + 1e-6);
                }
            }
        }
    }
}

TEST_CASE("GeoMap changes building modes at the same view using downloaded tiles", "[render][geomap][tiles]") {
    TileCache cache;
    const auto payload = WaterTile() + BuildingTile({
        {{1024, 1024}, {3072, 1024}, {3072, 3072}, {1024, 3072}},
    });
    auto source = std::make_unique<TileSource>(cache, payload);
    const auto* observed = source.get();
    GeoMapTiles tiles(std::move(source));
    auto config = TileConfig();
    for (const U64 mode : {0, 1, 2, 0, 2}) {
        CAPTURE(mode);
        tiles.stop();
        config.rendering.buildings = mode > 0;
        config.rendering.buildingOutlines = mode == 2;
        tiles.start(config);
        REQUIRE(WaitForWater(tiles, TileContext()));
        const auto& geometry = tiles.geometry();
        CHECK(geometry.fills[static_cast<U64>(GeoMapTiles::Fill::Building)].indices.empty() == (mode == 0));
        CHECK(geometry.lines[static_cast<U64>(GeoMapTiles::Line::Building)].empty() == (mode != 2));
        CHECK(observed->fetches.load() == 1);
    }
}

TEST_CASE("GeoMap rendering choices remove disabled tile layers and restore them from cache", "[render][geomap][tiles]") {
    const std::vector<U64> line{9, 2048, 2048, 10, 2048, 2048};
    const std::vector<U64> point{9, 4096, 4096};
    const auto payload = WaterTile() +
        LayerTile("transportation", 2, line, "primary") +
        LayerTile("transportation", 2, line, "rail") +
        LayerTile("waterway", 2, line, "river") +
        LayerTile("boundary", 2, line, "") +
        LayerTile("aeroway", 2, line, "runway") +
        LayerTile("aeroway", 3, {9, 2048, 2048, 26, 4096, 0, 0, 4096, 4095, 0, 15}, "apron") +
        LayerTile("place", 1, point, "city") +
        LayerTile("aerodrome_label", 1, point, "international");
    for (auto field : {&MapRenderingOptions::roads, &MapRenderingOptions::rivers,
                       &MapRenderingOptions::borders, &MapRenderingOptions::places,
                       &MapRenderingOptions::airports}) {
        TileCache cache;
        auto source = std::make_unique<TileSource>(cache, payload);
        const auto* observed = source.get();
        GeoMapTiles tiles(std::move(source));
        auto config = TileConfig();
        for (bool enabled : {true, false, true}) {
            tiles.stop();
            config.rendering.*field = enabled;
            tiles.start(config);
            REQUIRE(WaitForWater(tiles, TileContext()));
            const auto& g = tiles.geometry();
            auto hasLine = [&](GeoMapTiles::Line line) { return !g.lines[static_cast<U64>(line)].empty(); };
            CHECK(hasLine(GeoMapTiles::Line::PrimaryRoad) == config.rendering.roads);
            CHECK(hasLine(GeoMapTiles::Line::Rail) == config.rendering.roads);
            CHECK(hasLine(GeoMapTiles::Line::River) == config.rendering.rivers);
            CHECK(hasLine(GeoMapTiles::Line::CountryBorder) == config.rendering.borders);
            CHECK(hasLine(GeoMapTiles::Line::Runway) == config.rendering.airports);
            CHECK(g.fills[static_cast<U64>(GeoMapTiles::Fill::Airport)].indices.empty() == !config.rendering.airports);
            CHECK(g.labels.size() == (config.rendering.places ? 1u + config.rendering.airports : 0u));
            CHECK(observed->fetches.load() == 1);
        }
    }
}

TEST_CASE("GeoMap skips disabled geometry decoding and revalidates cached layers when enabled", "[render][geomap][tiles]") {
    TileCache cache;
    const auto context = TileContext();
    {
        const auto malformed = WaterTile() + LayerTile("transportation", 2, {9, 0}, "primary");
        GeoMapTiles tiles(std::make_unique<TileSource>(cache, malformed));
        auto config = TileConfig();
        config.rendering.roads = false;
        tiles.start(config);
        REQUIRE(WaitForWater(tiles, context));
        tiles.stop();
        REQUIRE(std::filesystem::exists(cache.tilePath()));
    }
    const auto valid = WaterTile() + LayerTile("transportation", 2, {9, 2048, 2048, 10, 2048, 2048}, "primary");
    auto source = std::make_unique<TileSource>(cache, valid);
    const auto* observed = source.get();
    GeoMapTiles tiles(std::move(source));
    tiles.start(TileConfig());
    REQUIRE(WaitForWater(tiles, context));
    REQUIRE_FALSE(tiles.geometry().lines[static_cast<U64>(GeoMapTiles::Line::PrimaryRoad)].empty());
    CHECK(observed->fetches.load() == 1);
}
