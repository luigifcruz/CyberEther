#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
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
