#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <zlib.h>
#include <nlohmann/json.hpp>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/geomap.hh"
#include "jetstream/render/components/text.hh"
#include "jetstream/render/components/font.hh"

#include "jetstream/types.hh"
#include "resources/shaders/map_shaders.hh"
#include "resources/geodata/geodata.hh"

namespace Jetstream::Render::Components {

static constexpr F32 kPi = static_cast<F32>(JST_PI);
static constexpr F32 MaxMercatorLatitude = 85.05112878f;

static inline F32 ClampMercatorLatitude(const F32 lat) {
    return std::clamp(lat, -MaxMercatorLatitude, MaxMercatorLatitude);
}

static inline F32 MercatorX(const F32 lon) {
    return (lon + 180.0f) / 360.0f;
}

static inline F32 WrapLongitude(const F32 longitude) {
    F32 wrapped = std::fmod(longitude + 180.0f, 360.0f);
    if (wrapped < 0.0f) {
        wrapped += 360.0f;
    }
    return wrapped - 180.0f;
}

static inline F32 WrapMercatorDelta(F32 delta) {
    if (delta > 0.5f) {
        delta -= 1.0f;
    } else if (delta < -0.5f) {
        delta += 1.0f;
    }
    return delta;
}

static inline F32 MercatorY(const F32 lat) {
    const F32 radians = ClampMercatorLatitude(lat) * kPi / 180.0f;
    const F32 projected =
        (1.0f - std::asinh(std::tan(radians)) / kPi) / 2.0f;
    return std::clamp(projected, 0.0f, 1.0f);
}

// Gzip decompression helper.

static bool DecompressGzip(const uint8_t* src,
                           uint32_t srcLen,
                           std::vector<uint8_t>& dst,
                           uint32_t rawLen) {
    dst.resize(rawLen);

    z_stream strm{};
    strm.next_in = const_cast<Bytef*>(src);
    strm.avail_in = srcLen;
    strm.next_out = dst.data();
    strm.avail_out = rawLen;

    if (inflateInit2(&strm, 15 + 16) != Z_OK) {
        return false;
    }

    int ret = inflate(&strm, Z_FINISH);
    inflateEnd(&strm);

    if (ret != Z_STREAM_END) {
        return false;
    }

    dst.resize(strm.total_out);
    return true;
}

// Static line binary loader.
// Binary format: [U32 float_count]
//                [F32 lon1,lat1,lon2,lat2 * segment_count]
//                [F32 x1,y1,x2,y2 * segment_count]

static Result LoadLineSegmentsFromMemory(const uint8_t* gz,
                                         uint32_t gzLen,
                                         uint32_t rawLen,
                                         std::vector<F32>& geographicVertices,
                                         std::vector<F32>& projectedVertices) {
    std::vector<uint8_t> raw;
    if (!DecompressGzip(gz, gzLen, raw, rawLen)) {
        JST_ERROR("[GEOMAP] Failed to decompress line data.");
        return Result::ERROR;
    }

    if (raw.size() < sizeof(U32)) {
        JST_ERROR("[GEOMAP] Line data is too small.");
        return Result::ERROR;
    }

    U32 floatCount;
    std::memcpy(&floatCount, raw.data(), sizeof(U32));
    if (floatCount == 0 || floatCount % 4 != 0) {
        JST_ERROR("[GEOMAP] Line data has an invalid size.");
        return Result::ERROR;
    }

    const U64 dataBytes = static_cast<U64>(floatCount) * sizeof(F32);
    if (dataBytes > (std::numeric_limits<U64>::max() - sizeof(U32)) / 2) {
        JST_ERROR("[GEOMAP] Line data size overflow.");
        return Result::ERROR;
    }
    const U64 expectedSize = sizeof(U32) + dataBytes * 2;
    if (raw.size() != expectedSize) {
        JST_ERROR("[GEOMAP] Line data size mismatch.");
        return Result::ERROR;
    }

    const U64 initialGeographicSize = geographicVertices.size();
    const U64 initialProjectedSize = projectedVertices.size();
    if (initialGeographicSize != initialProjectedSize) {
        JST_ERROR("[GEOMAP] Existing line coordinate streams are misaligned.");
        return Result::ERROR;
    }
    geographicVertices.resize(initialGeographicSize + floatCount);
    projectedVertices.resize(initialProjectedSize + floatCount);
    std::memcpy(geographicVertices.data() + initialGeographicSize,
                raw.data() + sizeof(U32),
                static_cast<size_t>(dataBytes));
    std::memcpy(projectedVertices.data() + initialProjectedSize,
                raw.data() + sizeof(U32) + dataBytes,
                static_cast<size_t>(dataBytes));

    for (U64 i = initialGeographicSize;
         i < geographicVertices.size(); i += 2) {
        const F32 lon = geographicVertices[i];
        const F32 lat = geographicVertices[i + 1];
        if (!std::isfinite(lon) || !std::isfinite(lat) ||
            lon < -180.0f || lon > 180.0f ||
            lat < -90.0f || lat > 90.0f) {
            geographicVertices.resize(initialGeographicSize);
            projectedVertices.resize(initialProjectedSize);
            JST_ERROR("[GEOMAP] Line data has an invalid lon/lat position.");
            return Result::ERROR;
        }
    }

    for (U64 i = initialProjectedSize;
         i < projectedVertices.size(); ++i) {
        const F32 position = projectedVertices[i];
        if (!std::isfinite(position) ||
            position < 0.0f || position > 1.0f) {
            geographicVertices.resize(initialGeographicSize);
            projectedVertices.resize(initialProjectedSize);
            JST_ERROR("[GEOMAP] Line data has an invalid Mercator position.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

// Pre-triangulated binary loader.
// Binary format: [U32 vertex_count][U32 index_count]
//                [F32 lon,lat * vertex_count]
//                [F32 mercator_x,mercator_y * vertex_count]
//                [U32 * index_count]

static Result LoadPreTriangulatedFromMemory(const uint8_t* gz,
                                            uint32_t gzLen,
                                            uint32_t rawLen,
                                            std::vector<F32>& geographicVertices,
                                            std::vector<F32>& projectedVertices,
                                            std::vector<U32>& indices) {
    std::vector<uint8_t> raw;
    if (!DecompressGzip(gz, gzLen, raw, rawLen)) {
        JST_ERROR("[GEOMAP] Failed to decompress "
                  "pre-triangulated data.");
        return Result::ERROR;
    }

    if (raw.size() < 8) {
        JST_ERROR("[GEOMAP] Pre-triangulated data too small.");
        return Result::ERROR;
    }

    const uint8_t* ptr = raw.data();
    U32 vertexCount;
    U32 indexCount;
    std::memcpy(&vertexCount, ptr, sizeof(U32));
    ptr += sizeof(U32);
    std::memcpy(&indexCount, ptr, sizeof(U32));
    ptr += sizeof(U32);

    if (vertexCount == 0 || indexCount == 0) {
        JST_ERROR("[GEOMAP] Pre-triangulated data is empty.");
        return Result::ERROR;
    }

    const U64 vertexBytes = static_cast<U64>(vertexCount) * 2 * sizeof(F32);
    const U64 indexBytes = static_cast<U64>(indexCount) * sizeof(U32);
    if (vertexBytes > (std::numeric_limits<U64>::max() - 8) / 2 ||
        indexBytes > std::numeric_limits<U64>::max() - 8 - vertexBytes * 2) {
        JST_ERROR("[GEOMAP] Pre-triangulated data size overflow.");
        return Result::ERROR;
    }

    const U64 expectedSize = 8 + vertexBytes * 2 + indexBytes;
    if (raw.size() != expectedSize || indexCount % 3 != 0) {
        JST_ERROR("[GEOMAP] Pre-triangulated data size "
                  "mismatch.");
        return Result::ERROR;
    }

    geographicVertices.resize(static_cast<U64>(vertexCount) * 2);
    std::memcpy(geographicVertices.data(), ptr,
                static_cast<size_t>(vertexBytes));
    ptr += vertexBytes;

    projectedVertices.resize(static_cast<U64>(vertexCount) * 2);
    std::memcpy(projectedVertices.data(), ptr,
                static_cast<size_t>(vertexBytes));
    ptr += vertexBytes;

    for (U64 i = 0; i < geographicVertices.size(); i += 2) {
        const F32 lon = geographicVertices[i];
        const F32 lat = geographicVertices[i + 1];
        if (!std::isfinite(lon) || !std::isfinite(lat) ||
            lon < -180.0f || lon > 180.0f ||
            lat < -90.0f || lat > 90.0f) {
            JST_ERROR("[GEOMAP] Pre-triangulated data has an invalid "
                      "lon/lat position.");
            geographicVertices.clear();
            projectedVertices.clear();
            return Result::ERROR;
        }
    }

    if (std::any_of(projectedVertices.begin(), projectedVertices.end(),
                    [](const F32 position) {
                        return !std::isfinite(position) ||
                               position < 0.0f || position > 1.0f;
                    })) {
        JST_ERROR("[GEOMAP] Pre-triangulated data has an invalid "
                  "Mercator position.");
        geographicVertices.clear();
        projectedVertices.clear();
        return Result::ERROR;
    }

    indices.resize(indexCount);
    std::memcpy(indices.data(), ptr, static_cast<size_t>(indexBytes));

    if (std::any_of(indices.begin(), indices.end(),
                    [vertexCount](const U32 index) {
                        return index >= vertexCount;
                    })) {
        JST_ERROR("[GEOMAP] Pre-triangulated data contains an invalid index.");
        geographicVertices.clear();
        projectedVertices.clear();
        indices.clear();
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

// Place label data.

struct PlaceInfo {
    F32 lon;
    F32 lat;
    F32 mercX;      // pre-computed Mercator X
    F32 mercY;      // pre-computed Mercator Y
    std::string name;
    I32 scalerank;
};

static Result LoadPlacesFromMemory(const uint8_t* gz,
                                   uint32_t gzLen,
                                   uint32_t rawLen,
                                   std::vector<PlaceInfo>& places) {
    std::vector<uint8_t> raw;
    if (!DecompressGzip(gz, gzLen, raw, rawLen)) {
        JST_ERROR("[GEOMAP] Failed to decompress places geodata.");
        return Result::ERROR;
    }

    nlohmann::json geojson;
    try {
        geojson = nlohmann::json::parse(raw.begin(), raw.end());
    } catch (const nlohmann::json::parse_error& e) {
        JST_ERROR("[GEOMAP] Failed to parse places GeoJSON: {}",
                  e.what());
        return Result::ERROR;
    }

    if (!geojson.contains("features") ||
        !geojson["features"].is_array()) {
        JST_ERROR("[GEOMAP] Places GeoJSON has no feature collection.");
        return Result::ERROR;
    }

    const U64 initialSize = places.size();
    try {
        for (const auto& feature : geojson["features"]) {
            if (!feature.contains("properties") ||
                !feature.contains("geometry")) {
                continue;
            }

            const auto& props = feature["properties"];
            const auto& geom = feature["geometry"];

            if (!props.contains("name") ||
                !props.contains("scalerank")) {
                continue;
            }

            // Prefer ASCII name for SDF text renderer (ASCII 32-127).
            std::string name;
            if (props.contains("nameascii") &&
                props["nameascii"].is_string()) {
                name = props["nameascii"].get<std::string>();
            } else {
                name = props["name"].get<std::string>();
            }
            if (name.empty()) {
                continue;
            }

            F32 lon = 0.0f;
            F32 lat = 0.0f;

            // Use geometry coordinates (Point type).
            if (geom.contains("coordinates") &&
                geom["coordinates"].is_array() &&
                geom["coordinates"].size() >= 2) {
                lon = geom["coordinates"][0].get<F32>();
                lat = geom["coordinates"][1].get<F32>();
            } else if (props.contains("longitude") &&
                       props.contains("latitude")) {
                lon = props["longitude"].get<F32>();
                lat = props["latitude"].get<F32>();
            } else {
                continue;
            }

            // Truncate long names at load time.
            if (name.size() > 19) {
                name = name.substr(0, 19);
            }

            const F32 mx = MercatorX(lon);
            const F32 my = MercatorY(lat);

            places.push_back({
                .lon = lon,
                .lat = lat,
                .mercX = mx,
                .mercY = my,
                .name = name,
                .scalerank = props["scalerank"].get<I32>(),
            });
        }
    } catch (const nlohmann::json::exception& e) {
        places.resize(initialSize);
        JST_ERROR("[GEOMAP] Failed to load place feature: {}", e.what());
        return Result::ERROR;
    }

    // Sort by scalerank ascending (most important first).
    std::sort(places.begin(), places.end(),
              [](const PlaceInfo& a, const PlaceInfo& b) {
                  return a.scalerank < b.scalerank;
              });

    JST_INFO("[GEOMAP] Loaded {} place labels.", places.size());
    return Result::SUCCESS;
}

// Internal GPU uniform struct (per-program).

struct GpuUniforms {
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
    float _pad0 = 0.0f;
    std::array<float, 4> worldOffsets{};
};

// Bathymetry depth layer descriptor.

struct BathymetrySource {
    const uint8_t* gz;
    uint32_t gzLen;
    uint32_t rawLen;
    U32 depth;       // meters
    float r, g, b;   // color for this depth level
};

// Color palette: 12 levels from shallow (0m) to deep (10000m).
// Lighter blue at surface, progressively darker navy at depth.
static constexpr BathymetrySource BathymetrySources[] = {
    {Resources::ne_10m_bathymetry_L_0_tri_gz,
     Resources::ne_10m_bathymetry_L_0_tri_gz_len,
     Resources::ne_10m_bathymetry_L_0_tri_raw_len,
     0,     0.106f, 0.176f, 0.310f},
    {Resources::ne_10m_bathymetry_K_200_tri_gz,
     Resources::ne_10m_bathymetry_K_200_tri_gz_len,
     Resources::ne_10m_bathymetry_K_200_tri_raw_len,
     200,   0.094f, 0.161f, 0.290f},
    {Resources::ne_10m_bathymetry_J_1000_tri_gz,
     Resources::ne_10m_bathymetry_J_1000_tri_gz_len,
     Resources::ne_10m_bathymetry_J_1000_tri_raw_len,
     1000,  0.082f, 0.145f, 0.271f},
    {Resources::ne_10m_bathymetry_I_2000_tri_gz,
     Resources::ne_10m_bathymetry_I_2000_tri_gz_len,
     Resources::ne_10m_bathymetry_I_2000_tri_raw_len,
     2000,  0.071f, 0.129f, 0.251f},
    {Resources::ne_10m_bathymetry_H_3000_tri_gz,
     Resources::ne_10m_bathymetry_H_3000_tri_gz_len,
     Resources::ne_10m_bathymetry_H_3000_tri_raw_len,
     3000,  0.059f, 0.114f, 0.231f},
    {Resources::ne_10m_bathymetry_G_4000_tri_gz,
     Resources::ne_10m_bathymetry_G_4000_tri_gz_len,
     Resources::ne_10m_bathymetry_G_4000_tri_raw_len,
     4000,  0.047f, 0.098f, 0.212f},
    {Resources::ne_10m_bathymetry_F_5000_tri_gz,
     Resources::ne_10m_bathymetry_F_5000_tri_gz_len,
     Resources::ne_10m_bathymetry_F_5000_tri_raw_len,
     5000,  0.039f, 0.082f, 0.192f},
    {Resources::ne_10m_bathymetry_E_6000_tri_gz,
     Resources::ne_10m_bathymetry_E_6000_tri_gz_len,
     Resources::ne_10m_bathymetry_E_6000_tri_raw_len,
     6000,  0.031f, 0.067f, 0.173f},
    {Resources::ne_10m_bathymetry_D_7000_tri_gz,
     Resources::ne_10m_bathymetry_D_7000_tri_gz_len,
     Resources::ne_10m_bathymetry_D_7000_tri_raw_len,
     7000,  0.024f, 0.055f, 0.153f},
    {Resources::ne_10m_bathymetry_C_8000_tri_gz,
     Resources::ne_10m_bathymetry_C_8000_tri_gz_len,
     Resources::ne_10m_bathymetry_C_8000_tri_raw_len,
     8000,  0.020f, 0.043f, 0.133f},
    {Resources::ne_10m_bathymetry_B_9000_tri_gz,
     Resources::ne_10m_bathymetry_B_9000_tri_gz_len,
     Resources::ne_10m_bathymetry_B_9000_tri_raw_len,
     9000,  0.016f, 0.035f, 0.114f},
    {Resources::ne_10m_bathymetry_A_10000_tri_gz,
     Resources::ne_10m_bathymetry_A_10000_tri_gz_len,
     Resources::ne_10m_bathymetry_A_10000_tri_raw_len,
     10000, 0.012f, 0.027f, 0.094f},
};

static constexpr U64 NumBathymetryLayers =
    sizeof(BathymetrySources) / sizeof(BathymetrySources[0]);

static constexpr bool BathymetryLayersAreOrdered() {
    for (U64 i = 1; i < NumBathymetryLayers; ++i) {
        if (BathymetrySources[i - 1].depth >= BathymetrySources[i].depth) {
            return false;
        }
    }
    return true;
}

static_assert(BathymetryLayersAreOrdered(),
              "Bathymetry layers must be ordered from shallow to deep.");

// Static quad vertices: 6 vertices forming 2 triangles.
// x = endpoint selector (0=start, 1=end), y = side offset (-1 or +1).
static const F32 QuadVertices[] = {
    0.0f, -1.0f,
    1.0f, -1.0f,
    0.0f,  1.0f,
    0.0f,  1.0f,
    1.0f, -1.0f,
    1.0f,  1.0f,
};

// GeoMap implementation.

GeoMap::GeoMap(const Config& config) {
    this->config = config;
    this->pimpl = std::make_unique<Impl>();
}

GeoMap::~GeoMap() {
    pimpl.reset();
}

// GPU resources for a merged colored fill layer.
struct MergedFillLayer {
    std::vector<F32> geographicPositions;  // longitude, latitude
    std::vector<F32> positions;            // exact Mercator x, y
    std::vector<F32> colors;               // r, g, b
    std::vector<U32> indices;
    U64 indexCount = 0;

    GpuUniforms gpuUniforms{};
    std::shared_ptr<Render::Buffer> posBuffer;     // Mercator x, y
    std::shared_ptr<Render::Buffer> colorBuffer;   // r, g, b
    std::shared_ptr<Render::Buffer> indexBuffer;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;
};

struct GeoMap::Impl {
    Uniforms uniforms;
    bool updateUniformsFlag = false;

    // Separate geodata categories.
    std::vector<F32> majorVertices;  // lon/lat coastline + country borders
    std::vector<F32> minorVertices;  // lon/lat state/province lines
    std::vector<F32> riverVertices;  // lon/lat rivers
    std::vector<F32> geographicLineVertices;
    // Exact flat-map coordinates generated alongside the source positions.
    std::vector<F32> majorProjectedVertices;
    std::vector<F32> minorProjectedVertices;
    std::vector<F32> riverProjectedVertices;
    std::vector<F32> geographicLineProjectedVertices;
    U64 majorInstanceCount = 0;
    U64 minorInstanceCount = 0;
    U64 riverInstanceCount = 0;
    U64 geographicLineInstanceCount = 0;

    // Merged fill layers (single draw call each).
    MergedFillLayer bathymetry;  // all 12 depth levels merged
    MergedFillLayer landcover;   // land + urban + lakes merged

    // GPU uniforms per program.
    GpuUniforms majorGpuUniforms{};
    GpuUniforms minorGpuUniforms{};
    GpuUniforms riverGpuUniforms{};
    GpuUniforms geographicLineGpuUniforms{};

    // Shared quad vertex buffer.
    std::shared_ptr<Render::Buffer> quadBuffer;

    // Major lines resources.
    std::shared_ptr<Render::Buffer> majorInstanceBuffer;
    std::shared_ptr<Render::Buffer> majorUniformBuffer;
    std::shared_ptr<Render::Vertex> majorVertex;
    std::shared_ptr<Render::Draw> majorDraw;
    std::shared_ptr<Render::Program> majorProgram;

    // Minor lines resources.
    std::shared_ptr<Render::Buffer> minorInstanceBuffer;
    std::shared_ptr<Render::Buffer> minorUniformBuffer;
    std::shared_ptr<Render::Vertex> minorVertex;
    std::shared_ptr<Render::Draw> minorDraw;
    std::shared_ptr<Render::Program> minorProgram;

    // River lines resources.
    std::shared_ptr<Render::Buffer> riverInstanceBuffer;
    std::shared_ptr<Render::Buffer> riverUniformBuffer;
    std::shared_ptr<Render::Vertex> riverVertex;
    std::shared_ptr<Render::Draw> riverDraw;
    std::shared_ptr<Render::Program> riverProgram;

    // Geographic reference lines resources.
    std::shared_ptr<Render::Buffer> geographicLineInstanceBuffer;
    std::shared_ptr<Render::Buffer> geographicLineUniformBuffer;
    std::shared_ptr<Render::Vertex> geographicLineVertex;
    std::shared_ptr<Render::Draw> geographicLineDraw;
    std::shared_ptr<Render::Program> geographicLineProgram;

    // Place labels.
    std::vector<PlaceInfo> places;
    std::shared_ptr<Render::Components::Text> text;
    static constexpr U64 LabelPoolSize = 48;
    static constexpr U64 LabelMaxChars = 20;
    std::array<std::string, LabelPoolSize> labelIds;
    U64 previousSlotCount = 0;
};

// Append triangulated polygons with one color to a merged layer.
static Result AppendColoredFill(const uint8_t* gz,
                                uint32_t gzLen,
                                uint32_t rawLen,
                                float r,
                                float g,
                                float b,
                                MergedFillLayer& layer) {
    std::vector<F32> tmpGeographic;
    std::vector<F32> tmpProjected;
    std::vector<U32> tmpIndices;
    JST_CHECK(LoadPreTriangulatedFromMemory(gz,
                                            gzLen,
                                            rawLen,
                                            tmpGeographic,
                                            tmpProjected,
                                            tmpIndices));

    const U64 vertexCount = tmpGeographic.size() / 2;
    const U64 currentVertexCount = layer.positions.size() / 2;
    if (tmpGeographic.size() != tmpProjected.size() ||
        layer.geographicPositions.size() != layer.positions.size()) {
        JST_ERROR("[GEOMAP] Fill coordinate streams are misaligned.");
        return Result::ERROR;
    }
    if (currentVertexCount > std::numeric_limits<U32>::max() ||
        vertexCount > std::numeric_limits<U32>::max() - currentVertexCount) {
        JST_ERROR("[GEOMAP] Merged fill layer has too many vertices.");
        return Result::ERROR;
    }
    if (layer.indices.size() > std::numeric_limits<U32>::max() ||
        tmpIndices.size() > std::numeric_limits<U32>::max() -
                            layer.indices.size()) {
        JST_ERROR("[GEOMAP] Merged fill layer has too many indices.");
        return Result::ERROR;
    }

    const U32 baseVertex = static_cast<U32>(currentVertexCount);
    layer.geographicPositions.reserve(
        layer.geographicPositions.size() + tmpGeographic.size());
    layer.positions.reserve(layer.positions.size() + tmpProjected.size());
    layer.colors.reserve(layer.colors.size() + vertexCount * 3);
    layer.indices.reserve(layer.indices.size() + tmpIndices.size());

    for (U64 i = 0; i < tmpGeographic.size(); i += 2) {
        layer.geographicPositions.push_back(tmpGeographic[i]);
        layer.geographicPositions.push_back(tmpGeographic[i + 1]);
        layer.positions.push_back(tmpProjected[i]);
        layer.positions.push_back(tmpProjected[i + 1]);
        layer.colors.push_back(r);
        layer.colors.push_back(g);
        layer.colors.push_back(b);
    }

    for (U32 idx : tmpIndices) {
        layer.indices.push_back(baseVertex + idx);
    }

    layer.indexCount = layer.indices.size();
    return Result::SUCCESS;
}

Result GeoMap::create(Window* window) {
    JST_INFO("[GEOMAP] Loading embedded coastline and provinces data.");

    // Load major lines: coastlines + country borders.
    JST_CHECK(LoadLineSegmentsFromMemory(
        Resources::ne_10m_coastline_segments_gz,
        Resources::ne_10m_coastline_segments_gz_len,
        Resources::ne_10m_coastline_segments_raw_len,
        pimpl->majorVertices,
        pimpl->majorProjectedVertices));

    JST_CHECK(LoadLineSegmentsFromMemory(
        Resources::ne_10m_admin_0_boundary_lines_land_segments_gz,
        Resources::ne_10m_admin_0_boundary_lines_land_segments_gz_len,
        Resources::ne_10m_admin_0_boundary_lines_land_segments_raw_len,
        pimpl->majorVertices,
        pimpl->majorProjectedVertices));

    // Load minor lines: state/province borders.
    JST_CHECK(LoadLineSegmentsFromMemory(
        Resources::ne_10m_admin_1_states_provinces_lines_segments_gz,
        Resources::ne_10m_admin_1_states_provinces_lines_segments_gz_len,
        Resources::ne_10m_admin_1_states_provinces_lines_segments_raw_len,
        pimpl->minorVertices,
        pimpl->minorProjectedVertices));

    // Load the equator, tropics, polar circles, and date line.
    JST_CHECK(LoadLineSegmentsFromMemory(
        Resources::ne_10m_geographic_lines_segments_gz,
        Resources::ne_10m_geographic_lines_segments_gz_len,
        Resources::ne_10m_geographic_lines_segments_raw_len,
        pimpl->geographicLineVertices,
        pimpl->geographicLineProjectedVertices));

    // These polygons are nested. Appending shallow to deep preserves painter
    // order while assigning exactly one color to every source layer.
    for (U64 i = 0; i < NumBathymetryLayers; ++i) {
        const auto& src = BathymetrySources[i];
        JST_CHECK(AppendColoredFill(src.gz,
                                    src.gzLen,
                                    src.rawLen,
                                    src.r,
                                    src.g,
                                    src.b,
                                    pimpl->bathymetry));
    }

    // Load land + urban + lakes into a single merged buffer.
    // Draw order: land first, urban on top, lakes on top.
    JST_CHECK(AppendColoredFill(Resources::ne_10m_land_tri_gz,
                                Resources::ne_10m_land_tri_gz_len,
                                Resources::ne_10m_land_tri_raw_len,
                                0.094f,
                                0.098f,
                                0.090f,  // land: dark gray-green
                                pimpl->landcover));

    JST_CHECK(AppendColoredFill(Resources::ne_10m_urban_areas_tri_gz,
                                Resources::ne_10m_urban_areas_tri_gz_len,
                                Resources::ne_10m_urban_areas_tri_raw_len,
                                0.133f,
                                0.133f,
                                0.122f,  // urban: slightly lighter
                                pimpl->landcover));

    JST_CHECK(AppendColoredFill(Resources::ne_10m_lakes_tri_gz,
                                Resources::ne_10m_lakes_tri_gz_len,
                                Resources::ne_10m_lakes_tri_raw_len,
                                0.106f,
                                0.176f,
                                0.310f,  // lakes: water blue
                                pimpl->landcover));

    // Load rivers (line data).
    JST_CHECK(LoadLineSegmentsFromMemory(
        Resources::ne_10m_rivers_lake_centerlines_segments_gz,
        Resources::ne_10m_rivers_lake_centerlines_segments_gz_len,
        Resources::ne_10m_rivers_lake_centerlines_segments_raw_len,
        pimpl->riverVertices,
        pimpl->riverProjectedVertices));

    // Flat rendering uses exact precomputed Mercator segment endpoints.
    pimpl->majorInstanceCount = pimpl->majorProjectedVertices.size() / 4;
    pimpl->minorInstanceCount = pimpl->minorProjectedVertices.size() / 4;
    pimpl->riverInstanceCount = pimpl->riverProjectedVertices.size() / 4;
    pimpl->geographicLineInstanceCount =
        pimpl->geographicLineProjectedVertices.size() / 4;

    const U64 totalInstances =
        pimpl->majorInstanceCount + pimpl->minorInstanceCount +
        pimpl->riverInstanceCount + pimpl->geographicLineInstanceCount;

    JST_INFO("[GEOMAP] Loaded {} major + {} minor + {} river + {} "
             "geographic line segments ({} total).",
             pimpl->majorInstanceCount, pimpl->minorInstanceCount,
             pimpl->riverInstanceCount,
             pimpl->geographicLineInstanceCount, totalInstances);

    JST_INFO("[GEOMAP] Merged bathymetry: {} triangles (1 draw call).",
             pimpl->bathymetry.indexCount / 3);

    JST_INFO("[GEOMAP] Merged landcover: {} triangles (1 draw call).",
             pimpl->landcover.indexCount / 3);

    if (totalInstances == 0 &&
        pimpl->bathymetry.indexCount == 0 &&
        pimpl->landcover.indexCount == 0) {
        JST_ERROR("[GEOMAP] No geometry found.");
        return Result::ERROR;
    }

    // Build shared quad vertex buffer (for line rendering).
    if (totalInstances > 0) {
        Render::Buffer::Config cfg;
        cfg.buffer = const_cast<F32*>(QuadVertices);
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 12;  // 6 vertices * 2 components
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->quadBuffer, cfg));
    }

    // Helper lambda to build one line category.
    auto buildCategory = [&](std::vector<F32>& instanceData,
                             U64 instanceCount,
                             GpuUniforms& gpuUniforms,
                             float lineWidth,
                             float colorR, float colorG, float colorB,
                             std::shared_ptr<Render::Buffer>& instanceBuffer,
                             std::shared_ptr<Render::Buffer>& uniformBuffer,
                             std::shared_ptr<Render::Vertex>& vertex,
                             std::shared_ptr<Render::Draw>& draw,
                             std::shared_ptr<Render::Program>& program) -> Result {

        if (instanceCount == 0) {
            return Result::SUCCESS;
        }

        // Initialize GPU uniforms.
        gpuUniforms.centerLon = pimpl->uniforms.centerLon;
        gpuUniforms.centerLat = pimpl->uniforms.centerLat;
        gpuUniforms.zoom = pimpl->uniforms.zoom;
        gpuUniforms.aspectRatio = pimpl->uniforms.aspectRatio;
        gpuUniforms.surfaceScale = pimpl->uniforms.surfaceScale;
        gpuUniforms.lineWidth = lineWidth;
        gpuUniforms.colorR = colorR;
        gpuUniforms.colorG = colorG;
        gpuUniforms.colorB = colorB;
        gpuUniforms.viewportWidth = pimpl->uniforms.viewportWidth;
        gpuUniforms.viewportHeight = pimpl->uniforms.viewportHeight;

        // Instance buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = instanceData.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = instanceData.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(instanceBuffer, cfg));
        }

        // Uniform buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &gpuUniforms;
            cfg.elementByteSize = sizeof(GpuUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(uniformBuffer, cfg));
        }

        // Vertex config: quad vertices + instance data.
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {pimpl->quadBuffer, 2},  // stride 2 (vec2)
            };
            cfg.instances = {
                {instanceBuffer, 4},  // stride 4 (vec4)
            };
            JST_CHECK(window->build(vertex, cfg));
        }

        // Draw config: triangles with instancing.
        {
            Render::Draw::Config cfg;
            cfg.buffer = vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            cfg.numberOfInstances = instanceCount;
            JST_CHECK(window->build(draw, cfg));
        }

        // Program.
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["geoline"];
            cfg.draws = {draw};
            cfg.buffers = {
                {uniformBuffer, Render::Program::Target::VERTEX |
                                Render::Program::Target::FRAGMENT},
            };
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(program, cfg));
        }

        return Result::SUCCESS;
    };

    // Build major lines (coastlines + borders): thick, bright.
    JST_CHECK(buildCategory(pimpl->majorProjectedVertices,
                            pimpl->majorInstanceCount,
                            pimpl->majorGpuUniforms,
                            5.0f,  // lineWidth
                            0.5f,
                            0.6f,
                            0.7f,  // color (light gray-blue)
                            pimpl->majorInstanceBuffer,
                            pimpl->majorUniformBuffer,
                            pimpl->majorVertex,
                            pimpl->majorDraw,
                            pimpl->majorProgram));

    // Build minor lines (state/province): thin, dimmer.
    JST_CHECK(buildCategory(pimpl->minorProjectedVertices,
                            pimpl->minorInstanceCount,
                            pimpl->minorGpuUniforms,
                            2.0f,  // lineWidth
                            0.35f,
                            0.42f,
                            0.49f,  // color (darker gray-blue)
                            pimpl->minorInstanceBuffer,
                            pimpl->minorUniformBuffer,
                            pimpl->minorVertex,
                            pimpl->minorDraw,
                            pimpl->minorProgram));

    // Build river lines: thin, water-colored.
    JST_CHECK(buildCategory(pimpl->riverProjectedVertices,
                            pimpl->riverInstanceCount,
                            pimpl->riverGpuUniforms,
                            2.0f,  // lineWidth
                            0.106f,
                            0.176f,
                            0.310f,  // color (same as lakes)
                            pimpl->riverInstanceBuffer,
                            pimpl->riverUniformBuffer,
                            pimpl->riverVertex,
                            pimpl->riverDraw,
                            pimpl->riverProgram));

    // Build geographic reference lines: thin and subdued.
    JST_CHECK(buildCategory(pimpl->geographicLineProjectedVertices,
                            pimpl->geographicLineInstanceCount,
                            pimpl->geographicLineGpuUniforms,
                            1.0f,
                            0.28f,
                            0.34f,
                            0.38f,
                            pimpl->geographicLineInstanceBuffer,
                            pimpl->geographicLineUniformBuffer,
                            pimpl->geographicLineVertex,
                            pimpl->geographicLineDraw,
                            pimpl->geographicLineProgram));

    // Helper lambda to build a merged fill layer pipeline.
    auto buildMergedFill = [&](MergedFillLayer& layer) -> Result {
        if (layer.indexCount == 0) {
            return Result::SUCCESS;
        }

        if (layer.geographicPositions.size() != layer.positions.size() ||
            layer.positions.size() % 2 != 0 ||
            layer.colors.size() % 3 != 0 ||
            layer.positions.size() / 2 != layer.colors.size() / 3) {
            JST_ERROR("[GEOMAP] Fill positions and colors are misaligned.");
            return Result::ERROR;
        }

        layer.gpuUniforms.centerLon = pimpl->uniforms.centerLon;
        layer.gpuUniforms.centerLat = pimpl->uniforms.centerLat;
        layer.gpuUniforms.zoom = pimpl->uniforms.zoom;
        layer.gpuUniforms.aspectRatio =
            pimpl->uniforms.aspectRatio;
        layer.gpuUniforms.surfaceScale =
            pimpl->uniforms.surfaceScale;
        layer.gpuUniforms.lineWidth = 0.0f;
        layer.gpuUniforms.viewportWidth =
            pimpl->uniforms.viewportWidth;
        layer.gpuUniforms.viewportHeight =
            pimpl->uniforms.viewportHeight;

        // Position buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = layer.positions.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = layer.positions.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(layer.posBuffer, cfg));
        }

        // Color buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = layer.colors.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = layer.colors.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(layer.colorBuffer, cfg));
        }

        // Index buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = layer.indices.data();
            cfg.elementByteSize = sizeof(U32);
            cfg.size = layer.indices.size();
            cfg.target =
                Render::Buffer::Target::VERTEX_INDICES;
            JST_CHECK(window->build(layer.indexBuffer, cfg));
        }

        // Uniform buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &layer.gpuUniforms;
            cfg.elementByteSize = sizeof(GpuUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(layer.uniformBuffer, cfg));
        }

        // Vertex config: position + color + indices.
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {layer.posBuffer, 2},    // location 0: vec2
                {layer.colorBuffer, 3},  // location 1: vec3
            };
            cfg.indices = layer.indexBuffer;
            JST_CHECK(window->build(layer.vertex, cfg));
        }

        // Draw config: indexed triangles.
        {
            Render::Draw::Config cfg;
            cfg.buffer = layer.vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            cfg.numberOfInstances = 1;
            JST_CHECK(window->build(layer.draw, cfg));
        }

        // Program using per-vertex color shader.
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["fill"];
            cfg.draws = {layer.draw};
            cfg.buffers = {
                {layer.uniformBuffer,
                 Render::Program::Target::VERTEX |
                 Render::Program::Target::FRAGMENT},
            };
            cfg.enableAlphaBlending = false;
            JST_CHECK(window->build(layer.program, cfg));
        }

        return Result::SUCCESS;
    };

    JST_CHECK(buildMergedFill(pimpl->bathymetry));
    JST_CHECK(buildMergedFill(pimpl->landcover));

    // Load place labels.
    JST_CHECK(LoadPlacesFromMemory(
        Resources::ne_10m_populated_places_simple_gz,
        Resources::ne_10m_populated_places_simple_gz_len,
        Resources::ne_10m_populated_places_simple_raw_len,
        pimpl->places));

    // Build Text component for labels.
    if (!pimpl->places.empty() && window->hasFont("default_mono")) {
        Render::Components::Text::Config cfg;
        cfg.maxCharacters =
            pimpl->LabelPoolSize * pimpl->LabelMaxChars;
        cfg.color = {1.0f, 1.0f, 1.0f, 0.85f};
        cfg.font = window->font("default_mono");
        cfg.sharpness = 0.5f;

        for (U64 i = 0; i < pimpl->LabelPoolSize; ++i) {
            const auto id = jst::fmt::format("l{:03d}", i);
            cfg.elements[id] = {
                .scale = 0.7f,
                .alignment = {1, 1},
            };
        }

        JST_CHECK(window->build(pimpl->text, cfg));
        JST_CHECK(window->bind(pimpl->text));

        // Pre-compute pool ID strings.
        for (U64 i = 0; i < pimpl->LabelPoolSize; ++i) {
            pimpl->labelIds[i] =
                jst::fmt::format("l{:03d}", i);
        }
    }

    return Result::SUCCESS;
}

Result GeoMap::destroy(Window* window) {
    if (pimpl->text) {
        JST_CHECK(window->unbind(pimpl->text));
    }

    return Result::SUCCESS;
}

Result GeoMap::surface(Render::Surface::Config& config) {
    // Bathymetry (single merged draw call).
    if (pimpl->bathymetry.indexCount > 0) {
        config.programs.push_back(pimpl->bathymetry.program);
    }

    // Landcover: land + urban + lakes (single merged draw call).
    if (pimpl->landcover.indexCount > 0) {
        config.programs.push_back(pimpl->landcover.program);
    }

    // Rivers render on top of landcover.
    if (pimpl->geographicLineInstanceCount > 0) {
        config.programs.push_back(pimpl->geographicLineProgram);
    }

    if (pimpl->riverInstanceCount > 0) {
        config.programs.push_back(pimpl->riverProgram);
    }

    if (pimpl->majorInstanceCount > 0) {
        config.programs.push_back(pimpl->majorProgram);
    }
    if (pimpl->minorInstanceCount > 0) {
        config.programs.push_back(pimpl->minorProgram);
    }

    // Labels render on top of everything.
    if (pimpl->text) {
        JST_CHECK(pimpl->text->surface(config));
    }

    return Result::SUCCESS;
}

Result GeoMap::present() {
    if (pimpl->updateUniformsFlag) {
        if (pimpl->majorInstanceCount > 0) {
            pimpl->majorGpuUniforms.centerLon =
                pimpl->uniforms.centerLon;
            pimpl->majorGpuUniforms.centerLat =
                pimpl->uniforms.centerLat;
            pimpl->majorGpuUniforms.zoom = pimpl->uniforms.zoom;
            pimpl->majorGpuUniforms.aspectRatio =
                pimpl->uniforms.aspectRatio;
            pimpl->majorGpuUniforms.surfaceScale =
                pimpl->uniforms.surfaceScale;
            pimpl->majorGpuUniforms.viewportWidth =
                pimpl->uniforms.viewportWidth;
            pimpl->majorGpuUniforms.viewportHeight =
                pimpl->uniforms.viewportHeight;
            pimpl->majorUniformBuffer->update();
        }

        if (pimpl->minorInstanceCount > 0) {
            pimpl->minorGpuUniforms.centerLon =
                pimpl->uniforms.centerLon;
            pimpl->minorGpuUniforms.centerLat =
                pimpl->uniforms.centerLat;
            pimpl->minorGpuUniforms.zoom = pimpl->uniforms.zoom;
            pimpl->minorGpuUniforms.aspectRatio =
                pimpl->uniforms.aspectRatio;
            pimpl->minorGpuUniforms.surfaceScale =
                pimpl->uniforms.surfaceScale;
            pimpl->minorGpuUniforms.viewportWidth =
                pimpl->uniforms.viewportWidth;
            pimpl->minorGpuUniforms.viewportHeight =
                pimpl->uniforms.viewportHeight;
            pimpl->minorUniformBuffer->update();
        }

        if (pimpl->riverInstanceCount > 0) {
            pimpl->riverGpuUniforms.centerLon =
                pimpl->uniforms.centerLon;
            pimpl->riverGpuUniforms.centerLat =
                pimpl->uniforms.centerLat;
            pimpl->riverGpuUniforms.zoom =
                pimpl->uniforms.zoom;
            pimpl->riverGpuUniforms.aspectRatio =
                pimpl->uniforms.aspectRatio;
            pimpl->riverGpuUniforms.surfaceScale =
                pimpl->uniforms.surfaceScale;
            pimpl->riverGpuUniforms.viewportWidth =
                pimpl->uniforms.viewportWidth;
            pimpl->riverGpuUniforms.viewportHeight =
                pimpl->uniforms.viewportHeight;
            pimpl->riverUniformBuffer->update();
        }

        if (pimpl->geographicLineInstanceCount > 0) {
            pimpl->geographicLineGpuUniforms.centerLon =
                pimpl->uniforms.centerLon;
            pimpl->geographicLineGpuUniforms.centerLat =
                pimpl->uniforms.centerLat;
            pimpl->geographicLineGpuUniforms.zoom =
                pimpl->uniforms.zoom;
            pimpl->geographicLineGpuUniforms.aspectRatio =
                pimpl->uniforms.aspectRatio;
            pimpl->geographicLineGpuUniforms.surfaceScale =
                pimpl->uniforms.surfaceScale;
            pimpl->geographicLineGpuUniforms.viewportWidth =
                pimpl->uniforms.viewportWidth;
            pimpl->geographicLineGpuUniforms.viewportHeight =
                pimpl->uniforms.viewportHeight;
            pimpl->geographicLineUniformBuffer->update();
        }

        // Update merged fill layer uniforms.
        auto updateFillUniforms = [&](MergedFillLayer& layer) -> Result {
            if (layer.indexCount > 0) {
                layer.gpuUniforms.centerLon =
                    pimpl->uniforms.centerLon;
                layer.gpuUniforms.centerLat =
                    pimpl->uniforms.centerLat;
                layer.gpuUniforms.zoom =
                    pimpl->uniforms.zoom;
                layer.gpuUniforms.aspectRatio =
                    pimpl->uniforms.aspectRatio;
                layer.gpuUniforms.surfaceScale =
                    pimpl->uniforms.surfaceScale;
                layer.gpuUniforms.viewportWidth =
                    pimpl->uniforms.viewportWidth;
                layer.gpuUniforms.viewportHeight =
                    pimpl->uniforms.viewportHeight;

                const F32 scale = std::pow(2.0f, pimpl->uniforms.zoom);
                const F32 centerX = MercatorX(pimpl->uniforms.centerLon);
                const F32 halfExtentX =
                    pimpl->uniforms.aspectRatio / (2.0f * scale);
                U64 worldCopyCount = 0;
                if (centerX - halfExtentX < 0.0f) {
                    layer.gpuUniforms.worldOffsets[worldCopyCount++] = -1.0f;
                }
                layer.gpuUniforms.worldOffsets[worldCopyCount++] = 0.0f;
                if (centerX + halfExtentX > 1.0f) {
                    layer.gpuUniforms.worldOffsets[worldCopyCount++] = 1.0f;
                }

                layer.uniformBuffer->update();
                JST_CHECK(layer.draw->updateInstanceCount(worldCopyCount));
            }
            return Result::SUCCESS;
        };

        JST_CHECK(updateFillUniforms(pimpl->bathymetry));
        JST_CHECK(updateFillUniforms(pimpl->landcover));

        // Update place labels.
        if (pimpl->text && !pimpl->places.empty()) {
            const Extent2D<F32> pixelSize = {
                (2.0f * pimpl->uniforms.surfaceScale) /
                    pimpl->uniforms.viewportWidth,
                (2.0f * pimpl->uniforms.surfaceScale) /
                    pimpl->uniforms.viewportHeight,
            };
            pimpl->text->updatePixelSize(pixelSize);

            const F32 scale =
                std::pow(2.0f, pimpl->uniforms.zoom);
            const F32 cx = MercatorX(pimpl->uniforms.centerLon);
            const F32 cy = MercatorY(pimpl->uniforms.centerLat);
            const F32 ar = pimpl->uniforms.aspectRatio;

            // Visibility threshold based on zoom level.
            const I32 maxRank = static_cast<I32>(pimpl->uniforms.zoom * 1.2f);

            U64 slot = 0;
            for (const auto& place : pimpl->places) {
                if (slot >= pimpl->LabelPoolSize) {
                    break;
                }

                // Places are sorted by scalerank — once
                // past threshold, all remaining are too.
                if (place.scalerank > maxRank) {
                    break;
                }

                // Project using pre-computed Mercator.
                const F32 ndcX =
                    WrapMercatorDelta(place.mercX - cx) * scale *
                    2.0f / ar;
                const F32 ndcY =
                    (cy - place.mercY) * scale * 2.0f;

                // Skip if outside viewport.
                if (std::abs(ndcX) > 1.1f ||
                    std::abs(ndcY) > 1.1f) {
                    continue;
                }

                // Smooth fade near threshold.
                const F32 rankDist =
                    static_cast<F32>(maxRank) -
                    static_cast<F32>(place.scalerank);
                const F32 fade = std::clamp(rankDist, 0.0f,
                                            1.0f);

                const auto& id = pimpl->labelIds[slot];

                auto element = pimpl->text->get(id);
                element.position = {ndcX, ndcY};
                element.scale = 0.8f * fade;
                if (element.fill != place.name) {
                    element.fill = place.name;
                }
                pimpl->text->update(id, element);

                ++slot;
            }

            // Only clear slots that were used last frame.
            for (U64 i = slot;
                 i < pimpl->previousSlotCount; ++i) {
                const auto& id = pimpl->labelIds[i];
                auto element = pimpl->text->get(id);
                element.fill = "";
                element.scale = 0.0f;
                pimpl->text->update(id, element);
            }
            pimpl->previousSlotCount = slot;
        }

        pimpl->updateUniformsFlag = false;
    }

    // Always present text (handles GPU buffer uploads).
    if (pimpl->text) {
        JST_CHECK(pimpl->text->present());
    }

    return Result::SUCCESS;
}

Result GeoMap::updateUniforms(const Uniforms& uniforms) {
    if (!std::isfinite(uniforms.centerLon) ||
        !std::isfinite(uniforms.centerLat) ||
        !std::isfinite(uniforms.zoom) ||
        uniforms.zoom < 0.0f || uniforms.zoom > 24.0f ||
        !std::isfinite(uniforms.surfaceScale) ||
        uniforms.surfaceScale <= 0.0f || uniforms.surfaceScale > 64.0f) {
        JST_ERROR("[GEOMAP] Map uniforms are outside valid ranges.");
        return Result::ERROR;
    }

    pimpl->uniforms = uniforms;
    pimpl->uniforms.centerLon = WrapLongitude(uniforms.centerLon);
    pimpl->uniforms.centerLat = ClampMercatorLatitude(uniforms.centerLat);
    pimpl->uniforms.viewportWidth =
        std::isfinite(uniforms.viewportWidth)
            ? std::max(uniforms.viewportWidth, 1.0f)
            : 1.0f;
    pimpl->uniforms.viewportHeight =
        std::isfinite(uniforms.viewportHeight)
            ? std::max(uniforms.viewportHeight, 1.0f)
            : 1.0f;
    if (!std::isfinite(uniforms.aspectRatio) ||
        uniforms.aspectRatio < std::numeric_limits<F32>::epsilon()) {
        pimpl->uniforms.aspectRatio =
            pimpl->uniforms.viewportWidth / pimpl->uniforms.viewportHeight;
    }
    pimpl->updateUniformsFlag = true;
    return Result::SUCCESS;
}

const GeoMap::Uniforms& GeoMap::getUniforms() const {
    return pimpl->uniforms;
}

}  // namespace Jetstream::Render::Components
