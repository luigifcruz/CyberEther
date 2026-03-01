#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
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

static constexpr F32 Pi = 3.14159265358979323846f;
static constexpr F32 MaxMercatorLatitude = 85.05112878f;

static inline F32 ClampMercatorLatitude(const F32 lat) {
    return std::clamp(lat, -MaxMercatorLatitude, MaxMercatorLatitude);
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

// GeoJSON parsing helpers.

static void ParseLineString(const nlohmann::json& coords,
                            std::vector<F32>& vertices) {
    for (U64 i = 0; i + 1 < coords.size(); ++i) {
        const auto& a = coords[i];
        const auto& b = coords[i + 1];

        vertices.push_back(a[0].get<F32>());
        vertices.push_back(a[1].get<F32>());
        vertices.push_back(b[0].get<F32>());
        vertices.push_back(b[1].get<F32>());
    }
}

static void ParsePolygon(const nlohmann::json& rings,
                         std::vector<F32>& vertices) {
    for (const auto& ring : rings) {
        ParseLineString(ring, vertices);
    }
}

static void ParseGeometry(const nlohmann::json& geometry,
                          std::vector<F32>& vertices) {
    if (geometry.is_null() || !geometry.is_object()) {
        return;
    }
    if (!geometry.contains("type") || !geometry.contains("coordinates")) {
        return;
    }
    if (geometry["coordinates"].is_null()) {
        return;
    }

    const auto type = geometry["type"].get<std::string>();
    const auto& coords = geometry["coordinates"];

    if (type == "LineString") {
        ParseLineString(coords, vertices);
    } else if (type == "MultiLineString") {
        for (const auto& line : coords) {
            ParseLineString(line, vertices);
        }
    } else if (type == "Polygon") {
        ParsePolygon(coords, vertices);
    } else if (type == "MultiPolygon") {
        for (const auto& polygon : coords) {
            ParsePolygon(polygon, vertices);
        }
    }
}

static void ParseGeoJson(const nlohmann::json& geojson,
                         std::vector<F32>& vertices) {
    if (geojson.contains("features") &&
        geojson["features"].is_array()) {
        for (const auto& feature : geojson["features"]) {
            if (!feature.contains("geometry")) {
                continue;
            }
            ParseGeometry(feature["geometry"], vertices);
        }
    } else {
        ParseGeometry(geojson, vertices);
    }
}

static void LoadGeoJsonFromMemory(const uint8_t* gz, uint32_t gzLen,
                                  uint32_t rawLen,
                                  std::vector<F32>& vertices) {
    std::vector<uint8_t> raw;
    if (!DecompressGzip(gz, gzLen, raw, rawLen)) {
        JST_ERROR("[GEOMAP] Failed to decompress embedded geodata.");
        return;
    }

    nlohmann::json geojson;
    try {
        geojson = nlohmann::json::parse(raw.begin(), raw.end());
    } catch (const nlohmann::json::parse_error& e) {
        JST_ERROR("[GEOMAP] Failed to parse embedded GeoJSON: {}",
                  e.what());
        return;
    }

    ParseGeoJson(geojson, vertices);
}

// Pre-triangulated binary loader.
// Binary format: [U32 vertex_count][U32 index_count]
//                [F32 lon,lat * vertex_count][U32 * index_count]

static void LoadPreTriangulatedFromMemory(const uint8_t* gz,
                                          uint32_t gzLen,
                                          uint32_t rawLen,
                                          std::vector<F32>& vertices,
                                          std::vector<U32>& indices) {
    std::vector<uint8_t> raw;
    if (!DecompressGzip(gz, gzLen, raw, rawLen)) {
        JST_ERROR("[GEOMAP] Failed to decompress "
                  "pre-triangulated data.");
        return;
    }

    if (raw.size() < 8) {
        JST_ERROR("[GEOMAP] Pre-triangulated data too small.");
        return;
    }

    const uint8_t* ptr = raw.data();
    U32 vertexCount;
    U32 indexCount;
    std::memcpy(&vertexCount, ptr, sizeof(U32));
    ptr += sizeof(U32);
    std::memcpy(&indexCount, ptr, sizeof(U32));
    ptr += sizeof(U32);

    const U64 expectedSize = 8 + vertexCount * 2 * sizeof(F32) +
                             indexCount * sizeof(U32);
    if (raw.size() < expectedSize) {
        JST_ERROR("[GEOMAP] Pre-triangulated data size "
                  "mismatch.");
        return;
    }

    vertices.resize(vertexCount * 2);
    std::memcpy(vertices.data(), ptr, vertexCount * 2 * sizeof(F32));
    ptr += vertexCount * 2 * sizeof(F32);

    indices.resize(indexCount);
    std::memcpy(indices.data(), ptr, indexCount * sizeof(U32));
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

static void LoadPlacesFromMemory(const uint8_t* gz,
                                 uint32_t gzLen,
                                 uint32_t rawLen,
                                 std::vector<PlaceInfo>& places) {
    std::vector<uint8_t> raw;
    if (!DecompressGzip(gz, gzLen, raw, rawLen)) {
        JST_ERROR("[GEOMAP] Failed to decompress places geodata.");
        return;
    }

    nlohmann::json geojson;
    try {
        geojson = nlohmann::json::parse(raw.begin(), raw.end());
    } catch (const nlohmann::json::parse_error& e) {
        JST_ERROR("[GEOMAP] Failed to parse places GeoJSON: {}",
                  e.what());
        return;
    }

    if (!geojson.contains("features") ||
        !geojson["features"].is_array()) {
        return;
    }

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

        const F32 mx = (lon + 180.0f) / 360.0f;
        const F32 r = ClampMercatorLatitude(lat) * Pi / 180.0f;
        const F32 my = (1.0f - std::log(std::tan(r) +
            1.0f / std::cos(r)) / Pi) / 2.0f;

        places.push_back({
            .lon = lon,
            .lat = lat,
            .mercX = mx,
            .mercY = my,
            .name = name,
            .scalerank = props["scalerank"].get<I32>(),
        });
    }

    // Sort by scalerank ascending (most important first).
    std::sort(places.begin(), places.end(),
              [](const PlaceInfo& a, const PlaceInfo& b) {
                  return a.scalerank < b.scalerank;
              });

    JST_INFO("[GEOMAP] Loaded {} place labels.", places.size());
}

// Internal GPU uniform struct (per-program).

struct GpuUniforms {
    float centerLon;
    float centerLat;
    float zoom;
    float aspectRatio;
    float lineWidth;
    float colorR;
    float colorG;
    float colorB;
    float viewportWidth;
    float viewportHeight;
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
static const BathymetrySource BathymetrySources[] = {
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
    std::vector<F32> vertices;   // stride 5: lon, lat, r, g, b
    std::vector<U32> indices;
    U64 indexCount = 0;

    GpuUniforms gpuUniforms;
    std::shared_ptr<Render::Buffer> posBuffer;     // lon, lat
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
    std::vector<F32> majorVertices;  // coastlines + country borders
    std::vector<F32> minorVertices;  // state/province lines
    std::vector<F32> riverVertices;  // rivers
    U64 majorInstanceCount = 0;
    U64 minorInstanceCount = 0;
    U64 riverInstanceCount = 0;

    // Merged fill layers (single draw call each).
    MergedFillLayer bathymetry;  // all 12 depth levels merged
    MergedFillLayer landcover;   // land + urban + lakes merged

    // GPU uniforms per program.
    GpuUniforms majorGpuUniforms;
    GpuUniforms minorGpuUniforms;
    GpuUniforms riverGpuUniforms;

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

    // Place labels.
    std::vector<PlaceInfo> places;
    std::shared_ptr<Render::Components::Text> text;
    static constexpr U64 LabelPoolSize = 48;
    static constexpr U64 LabelMaxChars = 20;
    std::array<std::string, LabelPoolSize> labelIds;
    U64 previousSlotCount = 0;
};

// Append triangulated polygons with a color to a merged layer.
// Vertices are interleaved: (lon, lat, r, g, b) per vertex.
static void AppendColoredFill(const uint8_t* gz,
                              uint32_t gzLen,
                              uint32_t rawLen,
                              float r,
                              float g,
                              float b,
                              MergedFillLayer& layer) {
    std::vector<F32> tmpVerts;
    std::vector<U32> tmpIndices;
    LoadPreTriangulatedFromMemory(gz,
                                  gzLen,
                                  rawLen,
                                  tmpVerts,
                                  tmpIndices);

    // tmpVerts has stride 2 (lon, lat). Convert to stride 5.
    const U32 baseVertex =
        static_cast<U32>(layer.vertices.size() / 5);

    for (U64 i = 0; i < tmpVerts.size(); i += 2) {
        layer.vertices.push_back(tmpVerts[i]);      // lon
        layer.vertices.push_back(tmpVerts[i + 1]);  // lat
        layer.vertices.push_back(r);
        layer.vertices.push_back(g);
        layer.vertices.push_back(b);
    }

    for (U32 idx : tmpIndices) {
        layer.indices.push_back(baseVertex + idx);
    }

    layer.indexCount = layer.indices.size();
}

Result GeoMap::create(Window* window) {
    JST_INFO("[GEOMAP] Loading embedded coastline and provinces data.");

    // Load major lines: coastlines + country borders.
    LoadGeoJsonFromMemory(Resources::ne_10m_coastline_gz,
                          Resources::ne_10m_coastline_gz_len,
                          Resources::ne_10m_coastline_raw_len,
                          pimpl->majorVertices);

    LoadGeoJsonFromMemory(Resources::ne_10m_admin_0_boundary_lines_land_gz,
                          Resources::ne_10m_admin_0_boundary_lines_land_gz_len,
                          Resources::ne_10m_admin_0_boundary_lines_land_raw_len,
                          pimpl->majorVertices);

    // Load minor lines: state/province borders.
    LoadGeoJsonFromMemory(Resources::ne_10m_admin_1_states_provinces_lines_gz,
                          Resources::ne_10m_admin_1_states_provinces_lines_gz_len,
                          Resources::ne_10m_admin_1_states_provinces_lines_raw_len,
                          pimpl->minorVertices);

    // Load all bathymetry depth layers into a single merged buffer.
    for (U64 i = 0; i < NumBathymetryLayers; ++i) {
        const auto& src = BathymetrySources[i];
        AppendColoredFill(src.gz,
                          src.gzLen,
                          src.rawLen,
                          src.r,
                          src.g,
                          src.b,
                          pimpl->bathymetry);
    }

    // Load land + urban + lakes into a single merged buffer.
    // Draw order: land first, urban on top, lakes on top.
    AppendColoredFill(Resources::ne_10m_land_tri_gz,
                      Resources::ne_10m_land_tri_gz_len,
                      Resources::ne_10m_land_tri_raw_len,
                      0.094f,
                      0.098f,
                      0.090f,  // land: dark gray-green
                      pimpl->landcover);

    AppendColoredFill(Resources::ne_10m_urban_areas_tri_gz,
                      Resources::ne_10m_urban_areas_tri_gz_len,
                      Resources::ne_10m_urban_areas_tri_raw_len,
                      0.133f,
                      0.133f,
                      0.122f,  // urban: slightly lighter
                      pimpl->landcover);

    AppendColoredFill(Resources::ne_10m_lakes_tri_gz,
                      Resources::ne_10m_lakes_tri_gz_len,
                      Resources::ne_10m_lakes_tri_raw_len,
                      0.106f,
                      0.176f,
                      0.310f,  // lakes: water blue
                      pimpl->landcover);

    // Load rivers (line data).
    LoadGeoJsonFromMemory(Resources::ne_10m_rivers_lake_centerlines_gz,
                          Resources::ne_10m_rivers_lake_centerlines_gz_len,
                          Resources::ne_10m_rivers_lake_centerlines_raw_len,
                          pimpl->riverVertices);

    // Each instance = 4 floats (lon1, lat1, lon2, lat2).
    pimpl->majorInstanceCount = pimpl->majorVertices.size() / 4;
    pimpl->minorInstanceCount = pimpl->minorVertices.size() / 4;
    pimpl->riverInstanceCount = pimpl->riverVertices.size() / 4;

    const U64 totalInstances =
        pimpl->majorInstanceCount + pimpl->minorInstanceCount +
        pimpl->riverInstanceCount;

    JST_INFO("[GEOMAP] Loaded {} major + {} minor + {} river "
             "line segments ({} total).",
             pimpl->majorInstanceCount, pimpl->minorInstanceCount,
             pimpl->riverInstanceCount, totalInstances);

    JST_INFO("[GEOMAP] Merged bathymetry: {} triangles (1 draw call).",
             pimpl->bathymetry.indexCount / 3);

    JST_INFO("[GEOMAP] Merged landcover: {} triangles (1 draw call).",
             pimpl->landcover.indexCount / 3);

    if (totalInstances == 0 &&
        pimpl->bathymetry.indexCount == 0 &&
        pimpl->landcover.indexCount == 0) {
        JST_WARN("[GEOMAP] No geometry found.");
        return Result::SUCCESS;
    }

    // Build shared quad vertex buffer (for line rendering).
    if (totalInstances > 0) {
        Render::Buffer::Config cfg;
        cfg.buffer = const_cast<F32*>(QuadVertices);
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 12;  // 6 vertices * 2 components
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->quadBuffer, cfg));
        JST_CHECK(window->bind(pimpl->quadBuffer));
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
            JST_CHECK(window->bind(instanceBuffer));
        }

        // Uniform buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &gpuUniforms;
            cfg.elementByteSize = sizeof(GpuUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(uniformBuffer, cfg));
            JST_CHECK(window->bind(uniformBuffer));
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
            cfg.shaders = ShadersPackage["map"];
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
    JST_CHECK(buildCategory(pimpl->majorVertices,
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
    JST_CHECK(buildCategory(pimpl->minorVertices,
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
    JST_CHECK(buildCategory(pimpl->riverVertices,
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

    // Helper lambda to build a merged fill layer pipeline.
    auto buildMergedFill = [&](MergedFillLayer& layer) -> Result {
        if (layer.indexCount == 0) {
            return Result::SUCCESS;
        }

        layer.gpuUniforms.centerLon = pimpl->uniforms.centerLon;
        layer.gpuUniforms.centerLat = pimpl->uniforms.centerLat;
        layer.gpuUniforms.zoom = pimpl->uniforms.zoom;
        layer.gpuUniforms.aspectRatio =
            pimpl->uniforms.aspectRatio;
        layer.gpuUniforms.lineWidth = 0.0f;
        layer.gpuUniforms.viewportWidth =
            pimpl->uniforms.viewportWidth;
        layer.gpuUniforms.viewportHeight =
            pimpl->uniforms.viewportHeight;

        // Separate position (lon, lat) and color (r, g, b) buffers
        // from interleaved stride-5 data.
        const U64 vertexCount = layer.vertices.size() / 5;
        std::vector<F32> posData(vertexCount * 2);
        std::vector<F32> colorData(vertexCount * 3);

        for (U64 i = 0; i < vertexCount; ++i) {
            posData[i * 2]     = layer.vertices[i * 5];
            posData[i * 2 + 1] = layer.vertices[i * 5 + 1];
            colorData[i * 3]     = layer.vertices[i * 5 + 2];
            colorData[i * 3 + 1] = layer.vertices[i * 5 + 3];
            colorData[i * 3 + 2] = layer.vertices[i * 5 + 4];
        }

        // Position buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = posData.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = posData.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(layer.posBuffer, cfg));
            JST_CHECK(window->bind(layer.posBuffer));
        }

        // Color buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = colorData.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = colorData.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(layer.colorBuffer, cfg));
            JST_CHECK(window->bind(layer.colorBuffer));
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
            JST_CHECK(window->bind(layer.indexBuffer));
        }

        // Uniform buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &layer.gpuUniforms;
            cfg.elementByteSize = sizeof(GpuUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(layer.uniformBuffer, cfg));
            JST_CHECK(window->bind(layer.uniformBuffer));
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

        // Free CPU-side vertex data after GPU upload.
        layer.vertices.clear();
        layer.vertices.shrink_to_fit();
        layer.indices.clear();
        layer.indices.shrink_to_fit();

        return Result::SUCCESS;
    };

    JST_CHECK(buildMergedFill(pimpl->bathymetry));
    JST_CHECK(buildMergedFill(pimpl->landcover));

    // Load place labels.
    LoadPlacesFromMemory(Resources::ne_10m_populated_places_simple_gz,
                         Resources::ne_10m_populated_places_simple_gz_len,
                         Resources::ne_10m_populated_places_simple_raw_len,
                         pimpl->places);

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
                0.7f,           // scale
                {0.0f, 0.0f},   // position
                {1, 1},          // alignment (center, center)
                0.0f,            // rotation
                "",              // fill (empty initially)
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
    const U64 totalInstances =
        pimpl->majorInstanceCount + pimpl->minorInstanceCount +
        pimpl->riverInstanceCount;

    if (totalInstances > 0) {
        JST_CHECK(window->unbind(pimpl->quadBuffer));

        if (pimpl->majorInstanceCount > 0) {
            JST_CHECK(window->unbind(pimpl->majorInstanceBuffer));
            JST_CHECK(window->unbind(pimpl->majorUniformBuffer));
        }

        if (pimpl->minorInstanceCount > 0) {
            JST_CHECK(window->unbind(pimpl->minorInstanceBuffer));
            JST_CHECK(window->unbind(pimpl->minorUniformBuffer));
        }

        if (pimpl->riverInstanceCount > 0) {
            JST_CHECK(window->unbind(pimpl->riverInstanceBuffer));
            JST_CHECK(window->unbind(pimpl->riverUniformBuffer));
        }
    }

    auto unbindMergedFill = [&](MergedFillLayer& layer) -> Result {
        if (layer.indexCount > 0) {
            JST_CHECK(window->unbind(layer.posBuffer));
            JST_CHECK(window->unbind(layer.colorBuffer));
            JST_CHECK(window->unbind(layer.indexBuffer));
            JST_CHECK(window->unbind(layer.uniformBuffer));
        }
        return Result::SUCCESS;
    };

    JST_CHECK(unbindMergedFill(pimpl->bathymetry));
    JST_CHECK(unbindMergedFill(pimpl->landcover));

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
            pimpl->riverGpuUniforms.viewportWidth =
                pimpl->uniforms.viewportWidth;
            pimpl->riverGpuUniforms.viewportHeight =
                pimpl->uniforms.viewportHeight;
            pimpl->riverUniformBuffer->update();
        }

        // Update merged fill layer uniforms.
        auto updateFillUniforms = [&](MergedFillLayer& layer) {
            if (layer.indexCount > 0) {
                layer.gpuUniforms.centerLon =
                    pimpl->uniforms.centerLon;
                layer.gpuUniforms.centerLat =
                    pimpl->uniforms.centerLat;
                layer.gpuUniforms.zoom =
                    pimpl->uniforms.zoom;
                layer.gpuUniforms.aspectRatio =
                    pimpl->uniforms.aspectRatio;
                layer.gpuUniforms.viewportWidth =
                    pimpl->uniforms.viewportWidth;
                layer.gpuUniforms.viewportHeight =
                    pimpl->uniforms.viewportHeight;
                layer.uniformBuffer->update();
            }
        };

        updateFillUniforms(pimpl->bathymetry);
        updateFillUniforms(pimpl->landcover);

        // Update place labels.
        if (pimpl->text && !pimpl->places.empty()) {
            const Extent2D<F32> pixelSize = {
                2.0f / pimpl->uniforms.viewportWidth,
                2.0f / pimpl->uniforms.viewportHeight,
            };
            pimpl->text->updatePixelSize(pixelSize);

            const F32 scale =
                std::pow(2.0f, pimpl->uniforms.zoom);
            const F32 r =
                ClampMercatorLatitude(pimpl->uniforms.centerLat) * Pi / 180.0f;
            const F32 cx =
                (pimpl->uniforms.centerLon + 180.0f) /
                360.0f;
            const F32 cy =
                (1.0f - std::log(std::tan(r) +
                    1.0f / std::cos(r)) / Pi) / 2.0f;
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
                    (place.mercX - cx) * scale *
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
                element.scale = 0.7f * fade;
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
    pimpl->uniforms = uniforms;
    pimpl->updateUniformsFlag = true;
    return Result::SUCCESS;
}

const GeoMap::Uniforms& GeoMap::getUniforms() const {
    return pimpl->uniforms;
}

}  // namespace Jetstream::Render::Components
