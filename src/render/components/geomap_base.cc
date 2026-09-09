#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <zlib.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/render/base.hh"
#include "geomap_base.hh"
#include "jetstream/render/components/text.hh"
#include "jetstream/render/components/font.hh"
#include "render/components/geomap_labels.hh"
#include "render/components/geomap_lines.hh"

#include "jetstream/types.hh"
#include "resources/shaders/map_shaders.hh"
#include "resources/geodata/geodata.hh"

namespace Jetstream::Render::Components {

static constexpr F32 kPi = static_cast<F32>(JST_PI);
static_assert(Resources::GeoDataFormatVersion == 2,
              "Rebuild geodata with the geographic-only binary format.");

static glm::vec3 LonLatToSphere(F32 lon, F32 lat) {
    return MapContext::LonLatToSphere(lon, lat);
}

static inline F32 WrapLongitude(const F32 longitude) {
    return MapContext::WrapLongitude(longitude);
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

static Result LoadLineSegmentsFromMemory(const uint8_t* gz,
                                         uint32_t gzLen,
                                         uint32_t rawLen,
                                         std::vector<F32>& geographicVertices) {
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
    const U64 expectedSize = sizeof(U32) + dataBytes;
    if (raw.size() != expectedSize) {
        JST_ERROR("[GEOMAP] Line data size mismatch.");
        return Result::ERROR;
    }

    const U64 initialGeographicSize = geographicVertices.size();
    geographicVertices.resize(initialGeographicSize + floatCount);
    std::memcpy(geographicVertices.data() + initialGeographicSize,
                raw.data() + sizeof(U32),
                static_cast<size_t>(dataBytes));

    for (U64 i = initialGeographicSize;
         i < geographicVertices.size(); i += 2) {
        const F32 lon = geographicVertices[i];
        const F32 lat = geographicVertices[i + 1];
        if (!std::isfinite(lon) || !std::isfinite(lat) ||
            lon < -180.0f || lon > 180.0f ||
            lat < -90.0f || lat > 90.0f) {
            geographicVertices.resize(initialGeographicSize);
            JST_ERROR("[GEOMAP] Line data has an invalid lon/lat position.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

struct StyledLineSegments {
    static constexpr U64 MaxCategories = 11;
    std::array<std::vector<F32>, MaxCategories> geographic;
    std::array<std::vector<F32>, MaxCategories> minZoom;
};

// Styled line binary format: [U32 segment_count]
//                            [F32 lon1,lat1,lon2,lat2 * segment_count]
//                            [F32 min_zoom * segment_count]
//                            [U8 class * segment_count]
static Result LoadStyledLineSegmentsFromMemory(const uint8_t* gz,
                                                uint32_t gzLen,
                                                uint32_t rawLen,
                                                U64 categoryCount,
                                                StyledLineSegments& output) {
    std::vector<uint8_t> raw;
    if (!DecompressGzip(gz, gzLen, raw, rawLen)) {
        JST_ERROR("[GEOMAP] Failed to decompress styled line data.");
        return Result::ERROR;
    }
    if (raw.size() < sizeof(U32)) {
        JST_ERROR("[GEOMAP] Styled line data is too small.");
        return Result::ERROR;
    }

    U32 segmentCount;
    std::memcpy(&segmentCount, raw.data(), sizeof(U32));
    const U64 coordinateBytes = static_cast<U64>(segmentCount) * 4 * sizeof(F32);
    const U64 zoomBytes = static_cast<U64>(segmentCount) * sizeof(F32);
    const U64 expectedSize = sizeof(U32) + coordinateBytes + zoomBytes +
                             segmentCount;
    if (segmentCount == 0 || raw.size() != expectedSize) {
        JST_ERROR("[GEOMAP] Styled line data size mismatch.");
        return Result::ERROR;
    }

    const uint8_t* geographic = raw.data() + sizeof(U32);
    const uint8_t* minZoom = geographic + coordinateBytes;
    const uint8_t* classes = minZoom + zoomBytes;
    for (U64 i = 0; i < segmentCount; ++i) {
        const U8 category = classes[i];
        if (category >= categoryCount ||
            category >= output.geographic.size()) {
            JST_ERROR("[GEOMAP] Styled line has an invalid category.");
            return Result::ERROR;
        }
        F32 geographicSegment[4];
        F32 segmentMinZoom;
        std::memcpy(geographicSegment, geographic + i * sizeof(geographicSegment),
                    sizeof(geographicSegment));
        std::memcpy(&segmentMinZoom, minZoom + i * sizeof(F32), sizeof(F32));
        const bool invalidGeographic =
            !std::isfinite(geographicSegment[0]) ||
            !std::isfinite(geographicSegment[1]) ||
            !std::isfinite(geographicSegment[2]) ||
            !std::isfinite(geographicSegment[3]) ||
            geographicSegment[0] < -180.0f || geographicSegment[0] > 180.0f ||
            geographicSegment[2] < -180.0f || geographicSegment[2] > 180.0f ||
            geographicSegment[1] < -90.0f || geographicSegment[1] > 90.0f ||
            geographicSegment[3] < -90.0f || geographicSegment[3] > 90.0f;
        if (!std::isfinite(segmentMinZoom) || segmentMinZoom < 0.0f ||
            invalidGeographic) {
            JST_ERROR("[GEOMAP] Styled line contains invalid data.");
            return Result::ERROR;
        }
        output.geographic[category].insert(output.geographic[category].end(),
                                           std::begin(geographicSegment),
                                           std::end(geographicSegment));
        output.minZoom[category].push_back(segmentMinZoom);
    }
    return Result::SUCCESS;
}

// Pre-triangulated binary loader.
// Binary format: [U32 vertex_count][U32 index_count]
//                [F32 lon,lat * vertex_count]
//                [U32 * index_count]

static Result LoadPreTriangulatedFromMemory(const uint8_t* gz,
                                            uint32_t gzLen,
                                            uint32_t rawLen,
                                            std::vector<F32>& geographicVertices,
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
    const U64 expectedSize = 8 + vertexBytes + indexBytes;
    if (raw.size() != expectedSize || indexCount % 3 != 0) {
        JST_ERROR("[GEOMAP] Pre-triangulated data size "
                  "mismatch.");
        return Result::ERROR;
    }

    geographicVertices.resize(static_cast<U64>(vertexCount) * 2);
    std::memcpy(geographicVertices.data(), ptr,
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
            return Result::ERROR;
        }
    }

    indices.resize(indexCount);
    std::memcpy(indices.data(), ptr, static_cast<size_t>(indexBytes));

    if (std::any_of(indices.begin(), indices.end(),
                    [vertexCount](const U32 index) {
                        return index >= vertexCount;
                    })) {
        JST_ERROR("[GEOMAP] Pre-triangulated data contains an invalid index.");
        geographicVertices.clear();
        indices.clear();
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

// Label data.
//
// Binary format: [U32 record_count]
//                per record:
//                  [F32 lon, lat]
//                  [I32 scalerank][F32 minZoom][F32 maxZoom][I32 population]
//                  [F32 area]  (latitude-corrected square degrees, states only)
//                  [U8 kind][U8 flags][U8 nameLen][U8 pathPointCount]
//                  [char[nameLen] name]   (ASCII 32-126)
//                  [F32 lon,lat * pathPointCount]

using LabelKind = GeoMapLabels::Kind;

struct LabelPathPoint {
    F32 lon;
    F32 lat;
};
static_assert(sizeof(LabelPathPoint) == sizeof(F32) * 2);

struct LabelInfo {
    F32 lon;
    F32 lat;
    I32 scalerank;
    F32 minZoom;
    F32 maxZoom;
    I32 population;
    F32 area;
    LabelKind kind;
    U8 flags;  // bit0 capital, bit1 worldcity, bit2 megacity, bit3 adm0cap
    std::string name;
    std::vector<LabelPathPoint> path;
};

static void UppercaseAscii(std::string& s) {
    for (char& c : s) {
        if (c >= 'a' && c <= 'z') {
            c = static_cast<char>(c - 'a' + 'A');
        }
    }
}

static Result LoadLabelsFromMemory(const uint8_t* gz,
                                   uint32_t gzLen,
                                   uint32_t rawLen,
                                   std::vector<LabelInfo>& labels) {
    std::vector<uint8_t> raw;
    if (!DecompressGzip(gz, gzLen, raw, rawLen)) {
        JST_ERROR("[GEOMAP] Failed to decompress label data.");
        return Result::ERROR;
    }

    if (raw.size() < sizeof(U32)) {
        JST_ERROR("[GEOMAP] Label data is too small.");
        return Result::ERROR;
    }

    const U64 initialSize = labels.size();

    const uint8_t* ptr = raw.data();
    const uint8_t* end = raw.data() + raw.size();

    U32 count;
    std::memcpy(&count, ptr, sizeof(U32));
    ptr += sizeof(U32);

    labels.reserve(labels.size() + count);

    for (U32 i = 0; i < count; ++i) {
        if (end - ptr < 32) {
            labels.resize(initialSize);
            JST_ERROR("[GEOMAP] Label data is truncated.");
            return Result::ERROR;
        }

        LabelInfo lbl{};
        std::memcpy(&lbl.lon, ptr, 4);      ptr += 4;
        std::memcpy(&lbl.lat, ptr, 4);      ptr += 4;
        std::memcpy(&lbl.scalerank, ptr, 4); ptr += 4;
        std::memcpy(&lbl.minZoom, ptr, 4);  ptr += 4;
        std::memcpy(&lbl.maxZoom, ptr, 4);  ptr += 4;
        std::memcpy(&lbl.population, ptr, 4); ptr += 4;
        std::memcpy(&lbl.area, ptr, 4);     ptr += 4;

        U8 kind, flags, nameLen, pathPointCount;
        std::memcpy(&kind, ptr, 1);     ptr += 1;
        std::memcpy(&flags, ptr, 1);    ptr += 1;
        std::memcpy(&nameLen, ptr, 1);  ptr += 1;
        std::memcpy(&pathPointCount, ptr, 1); ptr += 1;

        const U64 pathBytes = static_cast<U64>(pathPointCount) *
                              sizeof(LabelPathPoint);
        if (static_cast<U64>(end - ptr) < nameLen + pathBytes) {
            labels.resize(initialSize);
            JST_ERROR("[GEOMAP] Label record is truncated.");
            return Result::ERROR;
        }

        lbl.name.assign(reinterpret_cast<const char*>(ptr), nameLen);
        ptr += nameLen;
        lbl.path.resize(pathPointCount);
        if (pathBytes > 0) {
            std::memcpy(lbl.path.data(), ptr, static_cast<size_t>(pathBytes));
            ptr += pathBytes;
        }

        if (!std::isfinite(lbl.lon) || !std::isfinite(lbl.lat) ||
            lbl.lon < -180.0f || lbl.lon > 180.0f ||
            lbl.lat < -90.0f || lbl.lat > 90.0f) {
            labels.resize(initialSize);
            JST_ERROR("[GEOMAP] Label has an invalid lon/lat position.");
            return Result::ERROR;
        }

        if (!std::isfinite(lbl.minZoom) || !std::isfinite(lbl.maxZoom) ||
            lbl.minZoom < 0.0f || lbl.maxZoom < 0.0f ||
            (lbl.maxZoom > 0.0f && lbl.maxZoom < lbl.minZoom)) {
            labels.resize(initialSize);
            JST_ERROR("[GEOMAP] Label has invalid zoom thresholds.");
            return Result::ERROR;
        }
        if (!std::isfinite(lbl.area) || lbl.area < 0.0f) {
            labels.resize(initialSize);
            JST_ERROR("[GEOMAP] Label has an invalid area.");
            return Result::ERROR;
        }
        for (const auto& point : lbl.path) {
            if (!std::isfinite(point.lon) || !std::isfinite(point.lat) ||
                point.lon < -180.0f || point.lon > 180.0f ||
                point.lat < -90.0f || point.lat > 90.0f) {
                labels.resize(initialSize);
                JST_ERROR("[GEOMAP] Label has an invalid path point.");
                return Result::ERROR;
            }
        }

        lbl.kind = static_cast<LabelKind>(kind);
        lbl.flags = flags;
        labels.push_back(std::move(lbl));
    }

    return Result::SUCCESS;
}

// GPU uniform block shared by every map program. Alias the public layout
// declared in the header so the struct name reads cleanly here.
using GpuUniforms = MapContext::GpuUniforms;

static_assert(sizeof(GpuUniforms) == 144,
              "GpuUniforms must match the std140 ShaderUniforms block.");

// Uniform block for the background starfield. The stars live on the celestial
// sphere and are transformed by `skyViewProjection` (camera translation
// stripped), so they stay fixed in world space and rotate opposite to panning.
struct StarsUniforms {
    glm::mat4 skyViewProjection;
    float viewportWidth;
    float viewportHeight;
    float surfaceScale;
    float _pad;
};
static_assert(sizeof(StarsUniforms) == 80,
              "StarsUniforms must match the std140 StarsUniforms block.");

// Copy the camera/screen fields from the canonical camera block into a
// per-program uniform buffer, leaving the line-style fields untouched.
static void CopyCameraFields(GpuUniforms& dst, const GpuUniforms& src) {
    dst.viewProjection = src.viewProjection;
    dst.cameraPos = src.cameraPos;
    dst.targetNormal = src.targetNormal;
    dst.surfaceScale = src.surfaceScale;
    dst.viewportWidth = src.viewportWidth;
    dst.viewportHeight = src.viewportHeight;
}

// Bathymetry depth layer descriptor.

struct BathymetrySource {
    const uint8_t* gz;
    uint32_t gzLen;
    uint32_t rawLen;
    U32 depth;       // meters
    float r, g, b;   // color for this depth level
};

// Color palette: 11 levels from 200m to deep (10000m).
// Lighter blue at shelf depth, progressively darker navy at depth. The
// shallowest level (0-200m shelf) is covered by the procedural water sphere
// underlay, which provides the base ocean color with a guaranteed
// hole-free silhouette (see GenerateWaterSphereMesh below).
static constexpr BathymetrySource BathymetrySources[] = {
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

static const F32 MarkerQuadVertices[] = {
    -1.0f, -1.0f,
     1.0f, -1.0f,
    -1.0f,  1.0f,
    -1.0f,  1.0f,
     1.0f, -1.0f,
     1.0f,  1.0f,
};

// GeoMap implementation.

GeoMapBaseLayer::GeoMapBaseLayer() {
    this->pimpl = std::make_unique<Impl>();
}

GeoMapBaseLayer::~GeoMapBaseLayer() {
    pimpl.reset();
}

// GPU resources for a merged colored fill layer.
struct MergedFillLayer {
    std::vector<F32> geographicPositions;  // longitude, latitude
    std::vector<F32> colors;               // r, g, b
    std::vector<U32> indices;
    U64 indexCount = 0;

    GpuUniforms gpuUniforms{};
    std::shared_ptr<Render::Buffer> posBuffer;     // longitude, latitude
    std::shared_ptr<Render::Buffer> colorBuffer;   // r, g, b
    std::shared_ptr<Render::Buffer> indexBuffer;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;
};

// GPU resources for one instanced line category.
struct LineCategory {
    std::vector<F32> vertices;          // lon/lat segment endpoints
    std::vector<F32> minZooms;          // optional per-segment visibility
    std::vector<F32> visibleVertices;
    std::vector<F32> dashRanges;
    U64 instanceCount = 0;
    F32 baseDashPhase = 0.0f;
    F32 fullVisibilityZoom = 0.0f;
    F32 fadeRange = 1.5f;
    GpuUniforms gpuUniforms{};
    std::shared_ptr<Render::Buffer> instanceBuffer;
    std::shared_ptr<Render::Buffer> dashBuffer;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;
};

struct LineStyle {
    F32 width;
    std::array<F32, 3> color;
    F32 pattern = 0.0f;
    F32 dashScale = 0.0f;
    F32 dashPhase = 0.0f;
    F32 opacity = 1.0f;
};

struct PointMarkerLayer {
    // Instances contain longitude, latitude, radius in pixels, and opacity.
    std::vector<F32> instances;
    GpuUniforms gpuUniforms{};
    std::shared_ptr<Render::Buffer> instanceBuffer;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;
};

struct GeoMapBaseLayer::Impl {
    Uniforms uniforms;
    Tuning tuning;
    bool firstFrame = true;

    // Shared camera snapshot; this layer adds only cartographic styling.
    GpuUniforms cameraUniforms{};

    // Celestial-sphere view-projection (camera translation stripped) for the
    // background starfield.
    glm::mat4 skyViewProjection{1.0f};

    // Background starfield resources.
    struct {
        StarsUniforms uniforms{};
        std::vector<F32> instances;  // dir.xyz, brightness per star
        U64 count = 0;
        std::shared_ptr<Render::Buffer> instanceBuffer;
        std::shared_ptr<Render::Buffer> uniformBuffer;
        std::shared_ptr<Render::Vertex> vertex;
        std::shared_ptr<Render::Draw> draw;
        std::shared_ptr<Render::Program> program;
    } stars;

    LineCategory coastlines;
    LineCategory stateBorders;

    // Geographic reference lines split by style:
    // 0=equator (solid), 1=tropics (dashed), 2=polar (dashed),
    // 3=International Date Line (solid, thick).
    std::array<LineCategory, 4> geo{};
    static constexpr U64 GeoEquator = 0;
    static constexpr U64 GeoTropics = 1;
    static constexpr U64 GeoPolar = 2;
    static constexpr U64 GeoDateLine = 3;

    // River geometry split into Natural Earth scaleranks 0 through 10.
    std::array<LineCategory, 11> rivers{};

    // Country borders are separate from coastlines so their weight can differ.
    LineCategory countryBorders;

    // Disputed boundary classes: breakaway, claim, elusive, reference.
    std::array<LineCategory, 4> disputed{};

    // Merged fill layers (single draw call each).
    MergedFillLayer waterSphere;  // procedural base ocean sphere
    MergedFillLayer bathymetry;   // depth levels 200m..10000m merged
    MergedFillLayer landcover;    // land + urban + lakes merged

    // Translucent atmosphere reuses the water sphere's position/index mesh.
    struct {
        GpuUniforms gpuUniforms{};
        std::shared_ptr<Render::Buffer> uniformBuffer;
        std::shared_ptr<Render::Vertex> vertex;
        std::shared_ptr<Render::Draw> draw;
        std::shared_ptr<Render::Program> program;
    } atmosphere;

    // Lines use endpoint-selector quads; markers/stars use centered quads.
    std::shared_ptr<Render::Buffer> quadBuffer;
    std::shared_ptr<Render::Buffer> markerQuadBuffer;
    std::vector<F32> solidDashRanges;
    std::shared_ptr<Render::Buffer> solidDashBuffer;

    PointMarkerLayer worldCityMarkers;
    PointMarkerLayer capitalDotMarkers;
    PointMarkerLayer cityDotMarkers;
    PointMarkerLayer airportMarkers;

    // Typographically distinct label layers.
    struct LabelLayer {
        std::shared_ptr<Render::Components::Text> text;
        std::vector<LabelInfo> labels;
        U64 poolSize = 0;
        F32 baseScale = 1.0f;
        F32 visibilityOffset = 0.0f;
        F32 fadeRange = 1.5f;
        F32 collisionPaddingPixels = 3.0f;
        F32 minZoomFloor = 0.0f;  // absolute zoom below which nothing shows
        F32 maxZoomFloor = 0.0f;
        // Fraction of the way from the globe's horizon (0) to the sub-camera
        // point (1) below which a label fades out, so names are not squeezed
        // against the limb. Zero disables the fade.
        F32 limbFadeStart = 0.0f;
        F32 limbFadeRange = 0.2f;
        bool uppercase = false;
        I32 priority = 0;  // lower = drawn first (wins collision)
        std::vector<std::string> labelIds;
        U64 previousSlotCount = 0;
    };
    static constexpr U64 LabelCountries = 0;
    static constexpr U64 LabelCapitals = 1;
    static constexpr U64 LabelCities = 2;
    static constexpr U64 LabelStates = 3;
    static constexpr U64 LabelRegions = 4;
    static constexpr U64 LabelMarine = 5;
    static constexpr U64 LabelAirports = 6;
    static constexpr U64 LabelPhysical = 7;
    static constexpr U64 NumLabelLayers = 8;
    std::array<LabelLayer, NumLabelLayers> labelLayers{};

    // Per-glyph text pool for labels that follow a geographic feature path,
    // shared by mountain range names and the reference line names.
    static constexpr U64 PathLabelMaxGlyphs = 512;
    static constexpr U64 ReferenceLabelGlyphs = 320;
    std::shared_ptr<Render::Components::Text> pathText;
    std::vector<std::string> pathLabelIds;
    U64 previousPathGlyphCount = 0;
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
    std::vector<U32> tmpIndices;
    JST_CHECK(LoadPreTriangulatedFromMemory(gz,
                                            gzLen,
                                            rawLen,
                                            tmpGeographic,
                                            tmpIndices));

    const U64 vertexCount = tmpGeographic.size() / 2;
    const U64 currentVertexCount = layer.geographicPositions.size() / 2;
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
    layer.colors.reserve(layer.colors.size() + vertexCount * 3);
    layer.indices.reserve(layer.indices.size() + tmpIndices.size());

    for (U64 i = 0; i < tmpGeographic.size(); i += 2) {
        layer.geographicPositions.push_back(tmpGeographic[i]);
        layer.geographicPositions.push_back(tmpGeographic[i + 1]);
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

// Base ocean color (former bathymetry L_0 shallow-shelf blue).
static constexpr float kWaterSphereR = 0.106f;
static constexpr float kWaterSphereG = 0.176f;
static constexpr float kWaterSphereB = 0.310f;

// Fill a merged fill layer with a procedural lon/lat grid sphere painted in
// the base ocean color. Drawn underneath the bathymetry, it guarantees the
// globe always presents a complete, hole-free water silhouette: the source
// polygon triangulation contains long slivers whose straight 3D chord edges
// dip inside the sphere and would otherwise open crescent gaps near the
// limb where the starfield shows through the globe.
static void GenerateWaterSphereMesh(MergedFillLayer& layer) {
    constexpr U32 kLonStep = 2;   // degrees; chord sagitta ~1.5e-4 R
    constexpr U32 kLatStep = 2;
    constexpr U32 kLonCount = 360 / kLonStep + 1;  // 181 (seam duplicated)
    constexpr U32 kLatCount = 180 / kLatStep + 1;  // 91 (poles duplicated)

    layer.geographicPositions.reserve(kLonCount * kLatCount * 2);
    layer.colors.reserve(kLonCount * kLatCount * 3);
    layer.indices.reserve((kLonCount - 1) * (kLatCount - 1) * 6);

    for (U32 row = 0; row < kLatCount; ++row) {
        const F32 lat = -90.0f + static_cast<F32>(row * kLatStep);
        for (U32 col = 0; col < kLonCount; ++col) {
            const F32 lon = -180.0f + static_cast<F32>(col * kLonStep);
            layer.geographicPositions.push_back(lon);
            layer.geographicPositions.push_back(lat);
            layer.colors.push_back(kWaterSphereR);
            layer.colors.push_back(kWaterSphereG);
            layer.colors.push_back(kWaterSphereB);
        }
    }

    for (U32 row = 0; row < kLatCount - 1; ++row) {
        for (U32 col = 0; col < kLonCount - 1; ++col) {
            const U32 a = row * kLonCount + col;
            const U32 b = a + 1;
            const U32 c = a + kLonCount;
            const U32 d = c + 1;
            // Degenerate triangles at the poles rasterize to nothing.
            layer.indices.push_back(a);
            layer.indices.push_back(c);
            layer.indices.push_back(b);
            layer.indices.push_back(b);
            layer.indices.push_back(c);
            layer.indices.push_back(d);
        }
    }

    layer.indexCount = layer.indices.size();
}

Result GeoMapBaseLayer::create(Window* window, const MapContext& context) {
    JST_INFO("[GEOMAP] Loading embedded coastline and provinces data.");

    pimpl->uniforms = context.view;
    pimpl->cameraUniforms = context.camera;
    pimpl->skyViewProjection = context.skyViewProjection;

    // Load coastlines and country borders separately so they can use
    // independent visual weights.
    JST_CHECK(LoadLineSegmentsFromMemory(
        Resources::ne_10m_coastline_segments_gz,
        Resources::ne_10m_coastline_segments_gz_len,
        Resources::ne_10m_coastline_segments_raw_len,
        pimpl->coastlines.vertices));

    JST_CHECK(LoadLineSegmentsFromMemory(
        Resources::ne_10m_admin_0_boundary_lines_land_segments_gz,
        Resources::ne_10m_admin_0_boundary_lines_land_segments_gz_len,
        Resources::ne_10m_admin_0_boundary_lines_land_segments_raw_len,
        pimpl->countryBorders.vertices));
    pimpl->countryBorders.instanceCount =
        pimpl->countryBorders.vertices.size() / 4;

    // Load minor lines: state/province borders.
    JST_CHECK(LoadLineSegmentsFromMemory(
        Resources::ne_10m_admin_1_states_provinces_lines_segments_gz,
        Resources::ne_10m_admin_1_states_provinces_lines_segments_gz_len,
        Resources::ne_10m_admin_1_states_provinces_lines_segments_raw_len,
        pimpl->stateBorders.vertices));

    // Preserve disputed-boundary class and source minimum zoom metadata.
    {
        StyledLineSegments styled;
        JST_CHECK(LoadStyledLineSegmentsFromMemory(
            Resources::ne_10m_admin_0_boundary_lines_disputed_areas_styled_segments_gz,
            Resources::ne_10m_admin_0_boundary_lines_disputed_areas_styled_segments_gz_len,
            Resources::ne_10m_admin_0_boundary_lines_disputed_areas_styled_segments_raw_len,
            pimpl->disputed.size(),
            styled));
        for (U64 i = 0; i < pimpl->disputed.size(); ++i) {
            auto& category = pimpl->disputed[i];
            category.vertices = std::move(styled.geographic[i]);
            category.minZooms = std::move(styled.minZoom[i]);
            category.visibleVertices.resize(
                category.vertices.size(), 0.0f);
        }
    }

    // Load the geographic reference lines, split into four style groups:
    // equator, tropics, polar circles, and the International Date Line.
    {
        struct Group {
            U64 idx;
            const uint8_t* gz;
            uint32_t gzLen;
            uint32_t rawLen;
        };
        const Group groups[] = {
            {Impl::GeoEquator,
             Resources::ne_10m_geographic_lines_equator_segments_gz,
             Resources::ne_10m_geographic_lines_equator_segments_gz_len,
             Resources::ne_10m_geographic_lines_equator_segments_raw_len},
            {Impl::GeoTropics,
             Resources::ne_10m_geographic_lines_tropics_segments_gz,
             Resources::ne_10m_geographic_lines_tropics_segments_gz_len,
             Resources::ne_10m_geographic_lines_tropics_segments_raw_len},
            {Impl::GeoPolar,
             Resources::ne_10m_geographic_lines_polar_segments_gz,
             Resources::ne_10m_geographic_lines_polar_segments_gz_len,
             Resources::ne_10m_geographic_lines_polar_segments_raw_len},
            {Impl::GeoDateLine,
             Resources::ne_10m_geographic_lines_dateline_segments_gz,
             Resources::ne_10m_geographic_lines_dateline_segments_gz_len,
             Resources::ne_10m_geographic_lines_dateline_segments_raw_len},
        };
        for (const auto& g : groups) {
            JST_CHECK(LoadLineSegmentsFromMemory(
                g.gz, g.gzLen, g.rawLen,
                pimpl->geo[g.idx].vertices));
        }
    }

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

    // Load authoritative country polygons + urban + lakes into one buffer.
    // Draw order: countries first, urban on top, lakes on top.
    JST_CHECK(AppendColoredFill(Resources::ne_10m_admin_0_countries_tri_gz,
                                Resources::ne_10m_admin_0_countries_tri_gz_len,
                                Resources::ne_10m_admin_0_countries_tri_raw_len,
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

    // Load rivers split into all 11 source rank groups.
    {
        StyledLineSegments styled;
        JST_CHECK(LoadStyledLineSegmentsFromMemory(
            Resources::ne_10m_rivers_lake_centerlines_styled_segments_gz,
            Resources::ne_10m_rivers_lake_centerlines_styled_segments_gz_len,
            Resources::ne_10m_rivers_lake_centerlines_styled_segments_raw_len,
            pimpl->rivers.size(),
            styled));
        for (U64 i = 0; i < pimpl->rivers.size(); ++i) {
            auto& category = pimpl->rivers[i];
            category.vertices = std::move(styled.geographic[i]);
            category.instanceCount = category.vertices.size() / 4;
        }
    }

    pimpl->coastlines.instanceCount = pimpl->coastlines.vertices.size() / 4;
    pimpl->stateBorders.instanceCount = pimpl->stateBorders.vertices.size() / 4;
    for (auto& g : pimpl->geo) {
        g.instanceCount = g.vertices.size() / 4;
    }

    U64 geoInstanceCount = 0;
    for (const auto& g : pimpl->geo) {
        geoInstanceCount += g.instanceCount;
    }
    U64 disputedSourceCount = 0;
    for (const auto& category : pimpl->disputed) {
        disputedSourceCount += category.vertices.size() / 4;
    }
    U64 riverSourceCount = 0;
    for (const auto& category : pimpl->rivers) {
        riverSourceCount += category.instanceCount;
    }

    const U64 totalInstances =
        pimpl->coastlines.instanceCount + pimpl->stateBorders.instanceCount +
        pimpl->countryBorders.instanceCount +
        riverSourceCount + disputedSourceCount +
        geoInstanceCount;

    JST_INFO("[GEOMAP] Loaded {} coastline + {} country + {} state + {} river "
             "+ {} disputed + {} geographic line segments ({} total).",
              pimpl->coastlines.instanceCount, pimpl->countryBorders.instanceCount,
              pimpl->stateBorders.instanceCount, riverSourceCount,
              disputedSourceCount,
              geoInstanceCount, totalInstances);

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

    // Solid categories all read the same immutable zero stream. Only dashed
    // categories need their own per-segment distances and camera-time uploads.
    {
        U64 largest = std::max({pimpl->coastlines.vertices.size(),
                                pimpl->countryBorders.vertices.size(),
                                pimpl->stateBorders.vertices.size()});
        for (const auto* categories : {&pimpl->geo, &pimpl->disputed}) {
            for (const auto& category : *categories) {
                largest = std::max<U64>(largest, category.vertices.size());
            }
        }
        for (const auto& category : pimpl->rivers) {
            largest = std::max<U64>(largest, category.vertices.size());
        }
        if (largest > 0) {
            pimpl->solidDashRanges.resize(largest / 2, 0.0f);
            Render::Buffer::Config cfg;
            cfg.buffer = pimpl->solidDashRanges.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = pimpl->solidDashRanges.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(pimpl->solidDashBuffer, cfg));
        }
    }

    // Helper lambda to build one line category.
    auto buildCategory = [&](LineCategory& category, const LineStyle& style,
                             std::optional<U64> initialCount = std::nullopt) -> Result {
        auto& instanceData = category.visibleVertices.empty()
            ? category.vertices : category.visibleVertices;
        if (instanceData.empty()) {
            return Result::SUCCESS;
        }

        // Initialize GPU uniforms: camera/screen from the canonical camera
        // block, plus this category's line styling.
        auto& gpuUniforms = category.gpuUniforms;
        CopyCameraFields(gpuUniforms, pimpl->cameraUniforms);
        gpuUniforms.lineWidth = style.width;
        gpuUniforms.colorR = style.color[0];
        gpuUniforms.colorG = style.color[1];
        gpuUniforms.colorB = style.color[2];
        gpuUniforms.lineStyle = style.pattern;
        gpuUniforms.dashScale = style.dashScale;
        gpuUniforms.dashPhase = style.dashPhase;
        gpuUniforms.lineOpacity = style.opacity;

        // Instance buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = instanceData.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = instanceData.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(category.instanceBuffer, cfg));
        }

        // Uniform buffer.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &gpuUniforms;
            cfg.elementByteSize = sizeof(GpuUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(category.uniformBuffer, cfg));
        }

        if (style.pattern > 0.5f) {
            category.dashRanges.resize(instanceData.size() / 2, 0.0f);
            Render::Buffer::Config cfg;
            cfg.buffer = category.dashRanges.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = category.dashRanges.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(category.dashBuffer, cfg));
        } else {
            category.dashBuffer = pimpl->solidDashBuffer;
        }

        // Vertex config: quad vertices + instance data.
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {pimpl->quadBuffer, 2},  // stride 2 (vec2)
            };
            cfg.instances = {
                {category.instanceBuffer, 4},  // stride 4 (vec4)
                {category.dashBuffer, 2},
            };
            JST_CHECK(window->build(category.vertex, cfg));
        }

        // Draw config: triangles with instancing.
        {
            Render::Draw::Config cfg;
            cfg.buffer = category.vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            cfg.numberOfInstances = initialCount.value_or(category.instanceCount);
            JST_CHECK(window->build(category.draw, cfg));
        }

        // Program.
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["geoline"];
            cfg.draws = {category.draw};
            cfg.buffers = {
                {category.uniformBuffer, Render::Program::Target::VERTEX |
                                Render::Program::Target::FRAGMENT},
            };
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(category.program, cfg));
        }

        return Result::SUCCESS;
    };

    // Build coastlines: thick and bright.
    JST_CHECK(buildCategory(pimpl->coastlines, {5.0f, {0.5f, 0.6f, 0.7f}}));

    // Country borders are slightly thinner than coastlines.
    JST_CHECK(buildCategory(pimpl->countryBorders,
                            {.width = 3.5f, .color = {0.68f, 0.68f, 0.68f},
                             .opacity = 0.72f}));

    // Build minor lines (state/province): thin, dimmer.
    JST_CHECK(buildCategory(pimpl->stateBorders,
                            {.width = 3.0f, .color = {0.30f, 0.30f, 0.30f},
                             .opacity = 0.0f}));

    // Keep all 11 Natural Earth ranks. Ranks 0-3 are fully visible at zoom 0;
    // each subsequent rank fades in across the next zoom level.
    for (U64 rank = 0; rank < pimpl->rivers.size(); ++rank) {
        auto& river = pimpl->rivers[rank];
        const F32 rankMix = static_cast<F32>(rank) /
                            static_cast<F32>(pimpl->rivers.size() - 1);
        river.fullVisibilityZoom =
            GeoMapLabels::RiverFullVisibilityZoom(static_cast<I32>(rank));
        river.fadeRange = 1.0f;
        const F32 initialOpacity = GeoMapLabels::VisibilityFade(
            pimpl->uniforms.detailZoom,
            river.fullVisibilityZoom, river.fadeRange);
        JST_CHECK(buildCategory(river,
            {.width = GeoMapLabels::RiverLineWidth(static_cast<I32>(rank)),
             .color = {std::lerp(0.15f, 0.08f, rankMix),
                       std::lerp(0.27f, 0.16f, rankMix),
                       std::lerp(0.42f, 0.27f, rankMix)},
             .opacity = initialOpacity},
            initialOpacity > 0.0f ? river.instanceCount : 0));
    }

    // Build geographic reference lines, one category per style group.
    // Equator: solid, slightly brighter. Tropics + polar circles: dashed.
    // International Date Line: solid and thick.
    {
        struct GeoStyle {
            U64 idx;
            float lineWidth;
            float r, g, b;
            float lineStyle;
            float dashScale, dashPhase;
        };
        const GeoStyle styles[] = {
            {Impl::GeoEquator, 2.5f,  0.36f, 0.42f, 0.46f,
             0.0f, 0.0f, 0.0f},                             // solid, thick
            {Impl::GeoTropics, 2.0f,  0.30f, 0.36f, 0.40f,
             1.0f, 16.0f, 0.0f},                            // 16px dashes
            {Impl::GeoPolar,   2.0f,  0.30f, 0.36f, 0.40f,
             1.0f, 16.0f, 0.25f},                           // 16px dashes
            {Impl::GeoDateLine, 4.0f, 0.32f, 0.38f, 0.42f,
             0.0f, 0.0f, 0.0f},                             // solid, thick
        };
        for (const auto& s : styles) {
            auto& g = pimpl->geo[s.idx];
            g.baseDashPhase = s.dashPhase;
            JST_CHECK(buildCategory(g, {s.lineWidth, {s.r, s.g, s.b},
                                        s.lineStyle, s.dashScale,
                                        s.dashPhase}));
        }
    }

    // Build disputed boundary classes separately. Source minimum zooms are
    // applied when the dynamic instance buffers are populated in present().
    {
        struct DisputedStyle {
            float width;
            float r, g, b;
            float lineStyle;
            float dashScale, dashPhase;
        };
        const DisputedStyle styles[] = {
            {2.8f, 0.86f, 0.36f, 0.32f, 0.0f, 0.0f, 0.00f},
            {2.2f, 0.92f, 0.62f, 0.34f, 0.0f, 0.0f, 0.00f},
            {1.6f, 0.66f, 0.60f, 0.52f, 0.0f, 0.0f, 0.00f},
            {1.0f, 0.48f, 0.52f, 0.56f, 0.0f, 0.0f, 0.00f},
        };
        for (U64 i = 0; i < pimpl->disputed.size(); ++i) {
            auto& category = pimpl->disputed[i];
            const auto& style = styles[i];
            category.baseDashPhase = style.dashPhase;
            JST_CHECK(buildCategory(category,
                {style.width, {style.r, style.g, style.b}, style.lineStyle,
                 style.dashScale, style.dashPhase}, 0));
        }
    }

    // Helper lambda to build a merged fill layer pipeline.
    auto buildMergedFill = [&](MergedFillLayer& layer) -> Result {
        if (layer.indexCount == 0) {
            return Result::SUCCESS;
        }

        if (layer.geographicPositions.size() % 2 != 0 ||
            layer.colors.size() % 3 != 0 ||
            layer.geographicPositions.size() / 2 != layer.colors.size() / 3) {
            JST_ERROR("[GEOMAP] Fill positions and colors are misaligned.");
            return Result::ERROR;
        }

        CopyCameraFields(layer.gpuUniforms, pimpl->cameraUniforms);
        layer.gpuUniforms.lineWidth = 0.0f;

        // Position buffer: geographic (lon, lat) — the shader projects onto
        // the 3D globe.
        {
            Render::Buffer::Config cfg;
            cfg.buffer = layer.geographicPositions.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = layer.geographicPositions.size();
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

    GenerateWaterSphereMesh(pimpl->waterSphere);
    JST_CHECK(buildMergedFill(pimpl->waterSphere));
    JST_CHECK(buildMergedFill(pimpl->bathymetry));
    JST_CHECK(buildMergedFill(pimpl->landcover));

    // Atmosphere pass: reuse the procedural sphere mesh and inflate it in the
    // vertex shader so the soft halo extends beyond the planet silhouette.
    CopyCameraFields(pimpl->atmosphere.gpuUniforms, pimpl->cameraUniforms);
    pimpl->atmosphere.gpuUniforms.lineWidth =
        pimpl->tuning.atmosphereRadius;
    pimpl->atmosphere.gpuUniforms.colorR =
        pimpl->tuning.atmosphereColor[0];
    pimpl->atmosphere.gpuUniforms.colorG =
        pimpl->tuning.atmosphereColor[1];
    pimpl->atmosphere.gpuUniforms.colorB =
        pimpl->tuning.atmosphereColor[2];
    pimpl->atmosphere.gpuUniforms.dashScale =
        pimpl->tuning.atmosphereInnerStrength;
    pimpl->atmosphere.gpuUniforms.dashPhase =
        pimpl->tuning.atmosphereOuterStrength;
    pimpl->atmosphere.gpuUniforms.outlineStrength =
        pimpl->tuning.atmosphereOutlineStrength;
    {
        Render::Buffer::Config cfg;
        cfg.buffer = &pimpl->atmosphere.gpuUniforms;
        cfg.elementByteSize = sizeof(GpuUniforms);
        cfg.size = 1;
        cfg.target = Render::Buffer::Target::UNIFORM;
        JST_CHECK(window->build(pimpl->atmosphere.uniformBuffer, cfg));
    }
    {
        Render::Vertex::Config cfg;
        cfg.vertices = {
            {pimpl->waterSphere.posBuffer, 2},
        };
        cfg.indices = pimpl->waterSphere.indexBuffer;
        JST_CHECK(window->build(pimpl->atmosphere.vertex, cfg));
    }
    {
        Render::Draw::Config cfg;
        cfg.buffer = pimpl->atmosphere.vertex;
        cfg.mode = Render::Draw::Mode::TRIANGLES;
        cfg.numberOfInstances = 1;
        JST_CHECK(window->build(pimpl->atmosphere.draw, cfg));
    }
    {
        Render::Program::Config cfg;
        cfg.shaders = ShadersPackage["atmosphere"];
        cfg.draws = {pimpl->atmosphere.draw};
        cfg.buffers = {
            {pimpl->atmosphere.uniformBuffer,
             Render::Program::Target::VERTEX |
             Render::Program::Target::FRAGMENT},
        };
        cfg.enableAlphaBlending = true;
        JST_CHECK(window->build(pimpl->atmosphere.program, cfg));
    }

    // Load all label datasets into one buffer, then partition by kind into
    // typographically distinct label layers.
    std::vector<LabelInfo> allLabels;

    JST_CHECK(LoadLabelsFromMemory(
        Resources::ne_10m_populated_places_simple_labels_gz,
        Resources::ne_10m_populated_places_simple_labels_gz_len,
        Resources::ne_10m_populated_places_simple_labels_raw_len,
        allLabels));
    JST_CHECK(LoadLabelsFromMemory(
        Resources::ne_10m_admin_0_countries_labels_gz,
        Resources::ne_10m_admin_0_countries_labels_gz_len,
        Resources::ne_10m_admin_0_countries_labels_raw_len,
        allLabels));
    JST_CHECK(LoadLabelsFromMemory(
        Resources::ne_10m_admin_1_states_provinces_labels_gz,
        Resources::ne_10m_admin_1_states_provinces_labels_gz_len,
        Resources::ne_10m_admin_1_states_provinces_labels_raw_len,
        allLabels));
    JST_CHECK(LoadLabelsFromMemory(
        Resources::ne_10m_airports_labels_gz,
        Resources::ne_10m_airports_labels_gz_len,
        Resources::ne_10m_airports_labels_raw_len,
        allLabels));
    JST_CHECK(LoadLabelsFromMemory(
        Resources::ne_10m_geography_marine_polys_labels_gz,
        Resources::ne_10m_geography_marine_polys_labels_gz_len,
        Resources::ne_10m_geography_marine_polys_labels_raw_len,
        allLabels));
    JST_CHECK(LoadLabelsFromMemory(
        Resources::ne_10m_geography_regions_elevation_points_labels_gz,
        Resources::ne_10m_geography_regions_elevation_points_labels_gz_len,
        Resources::ne_10m_geography_regions_elevation_points_labels_raw_len,
        allLabels));
    JST_CHECK(LoadLabelsFromMemory(
        Resources::ne_10m_geography_regions_polys_labels_gz,
        Resources::ne_10m_geography_regions_polys_labels_gz_len,
        Resources::ne_10m_geography_regions_polys_labels_raw_len,
        allLabels));
    JST_CHECK(LoadLabelsFromMemory(
        Resources::ne_10m_rivers_lake_centerlines_labels_gz,
        Resources::ne_10m_rivers_lake_centerlines_labels_gz_len,
        Resources::ne_10m_rivers_lake_centerlines_labels_raw_len,
        allLabels));
    JST_CHECK(LoadLabelsFromMemory(
        Resources::ne_10m_lakes_labels_gz,
        Resources::ne_10m_lakes_labels_gz_len,
        Resources::ne_10m_lakes_labels_raw_len,
        allLabels));

    JST_INFO("[GEOMAP] Loaded {} total label candidates.", allLabels.size());

    // Cartographic hierarchy and visibility policy:
    //
    //   zoom -1.0     globe overview: top-ranked countries and oceans
    //        0.0-1.0  world cities and medium countries
    //        1.0-2.0  seas, small countries, national capitals
    //   -0.75-1.0    state borders ramp in gently over the globe view
    //        1.0-2.75 state names by size and rank (Texas-sized units first,
    //                 Länder-sized at 2.75) and local countries; smaller
    //                 administrative units follow
    //        2.5-3.5  megacities and major mountain ranges
    //        3.5-4.5  other capitals and cities above one million
    //        4.0-5.0  major airports and cities above 250k
    //        6.0-9.0  regional airports, towns, and detailed waterways
    //
    //   Natural Earth min_zoom values target web map zoom levels, which run
    //   about five levels ahead of detailZoom in this camera. City sources are
    //   pulled forward by citySourceLead and countries by countrySourceLead so
    //   the tiers above, not the source, decide when a class first appears.
    //
    // Lower priority values reserve collision space first and draw last:
    // country > state > capital > city > region > marine > airport >
    // physical/hydro. Candidate pools stay stable across zoom; semantic fades
    // and collision selection control density without batch-wise label pops.
    {
        auto& countries = pimpl->labelLayers[Impl::LabelCountries];
        countries.poolSize = 48;
        countries.baseScale = 1.12f;
        countries.uppercase = true;
        countries.fadeRange = 0.75f;
        countries.collisionPaddingPixels = 8.0f;
        countries.minZoomFloor = 0.0f;
        countries.maxZoomFloor = 8.0f;
        countries.limbFadeStart = 0.4f;
        countries.priority = 0;

        auto& capitals = pimpl->labelLayers[Impl::LabelCapitals];
        capitals.poolSize = 96;
        capitals.baseScale = 0.68f;
        capitals.visibilityOffset = 0.0f;
        capitals.fadeRange = 1.0f;
        capitals.collisionPaddingPixels = 5.0f;
        capitals.minZoomFloor = 0.0f;
        capitals.priority = 2;

        auto& cities = pimpl->labelLayers[Impl::LabelCities];
        cities.poolSize = 128;
        cities.baseScale = 0.68f;
        cities.visibilityOffset = 0.0f;
        cities.fadeRange = 1.0f;
        cities.collisionPaddingPixels = 4.0f;
        cities.minZoomFloor = 0.0f;
        cities.priority = 3;

        auto& states = pimpl->labelLayers[Impl::LabelStates];
        states.poolSize = 64;
        states.baseScale = 1.10f;
        states.uppercase = true;
        states.fadeRange =
            GeoMapLabels::StateLabelFullZoom -
            GeoMapLabels::StateLabelFadeStartZoom;
        states.minZoomFloor = GeoMapLabels::StateLabelFullZoom;
        states.priority = 1;

        auto& regions = pimpl->labelLayers[Impl::LabelRegions];
        regions.poolSize = 80;
        regions.baseScale = 0.72f;
        regions.fadeRange = 1.0f;
        regions.minZoomFloor = 0.0f;
        regions.priority = 4;

        auto& marine = pimpl->labelLayers[Impl::LabelMarine];
        marine.poolSize = 64;
        marine.baseScale = 0.82f;
        marine.uppercase = true;
        marine.fadeRange = 1.0f;
        marine.minZoomFloor = 0.0f;
        marine.priority = 5;

        auto& airports = pimpl->labelLayers[Impl::LabelAirports];
        airports.poolSize = 96;
        airports.baseScale = 0.68f;
        airports.uppercase = true;
        airports.fadeRange = 1.0f;
        airports.priority = 6;

        auto& physical = pimpl->labelLayers[Impl::LabelPhysical];
        physical.poolSize = 72;
        physical.baseScale = 0.70f;
        physical.fadeRange = 1.0f;
        physical.minZoomFloor = 0.0f;
        physical.priority = 7;
    }

    // Partition labels into their layers.
    for (const auto& lbl : allLabels) {
        switch (lbl.kind) {
            case LabelKind::Country:
                pimpl->labelLayers[Impl::LabelCountries].labels.push_back(lbl); break;
            case LabelKind::Capital:
                pimpl->labelLayers[Impl::LabelCapitals].labels.push_back(lbl); break;
            case LabelKind::City:
                pimpl->labelLayers[Impl::LabelCities].labels.push_back(lbl); break;
            case LabelKind::State:
                pimpl->labelLayers[Impl::LabelStates].labels.push_back(lbl); break;
            case LabelKind::Water:
                pimpl->labelLayers[Impl::LabelMarine].labels.push_back(lbl); break;
            case LabelKind::Region:
                pimpl->labelLayers[Impl::LabelRegions].labels.push_back(lbl); break;
            case LabelKind::Physical:
            case LabelKind::River:
            case LabelKind::Lake:
                pimpl->labelLayers[Impl::LabelPhysical].labels.push_back(lbl); break;
            case LabelKind::Airport:
                pimpl->labelLayers[Impl::LabelAirports].labels.push_back(lbl); break;
        }
    }

    // Sort each layer so the most important labels fill the pool first.
    {
        for (auto& layer : pimpl->labelLayers) {
            std::sort(layer.labels.begin(), layer.labels.end(),
                      GeoMapLabels::MoreImportant<LabelInfo>);
        }
    }

    JST_INFO("[GEOMAP] Labels: {} countries, {} capitals, {} cities, "
             "{} states, {} regions, {} marine, {} airports, {} physical/hydro.",
             pimpl->labelLayers[Impl::LabelCountries].labels.size(),
             pimpl->labelLayers[Impl::LabelCapitals].labels.size(),
             pimpl->labelLayers[Impl::LabelCities].labels.size(),
             pimpl->labelLayers[Impl::LabelStates].labels.size(),
             pimpl->labelLayers[Impl::LabelRegions].labels.size(),
             pimpl->labelLayers[Impl::LabelMarine].labels.size(),
             pimpl->labelLayers[Impl::LabelAirports].labels.size(),
             pimpl->labelLayers[Impl::LabelPhysical].labels.size());

    // Build a Text component for each non-empty layer.
    auto buildLabelLayer = [&](U64 idx, const char* fontName) -> Result {
        auto& layer = pimpl->labelLayers[idx];
        if (layer.labels.empty() || layer.poolSize == 0) {
            return Result::SUCCESS;
        }

        std::shared_ptr<Render::Components::Font> font;
        if (window->hasFont(fontName)) {
            font = window->font(fontName);
        } else if (window->hasFont("default_mono")) {
            font = window->font("default_mono");
        } else {
            return Result::SUCCESS;  // no font available; skip this layer
        }

        layer.labelIds.resize(layer.poolSize);

        Render::Components::Text::Config cfg;
        // Reserve for the full source names; country and state names can be long.
        cfg.maxCharacters = 1;
        for (const auto& label : layer.labels) {
            cfg.maxCharacters = std::max<U64>(cfg.maxCharacters, label.name.size());
        }
        cfg.color = {1.0f, 1.0f, 1.0f, 0.85f};
        cfg.font = font;
        cfg.sharpness = 0.5f;

        for (U64 i = 0; i < layer.poolSize; ++i) {
            const auto id = jst::fmt::format("l{:03d}", i);
            cfg.elements[id] = {
                .scale = 0.0f,
                .alignment = {1, 1},
            };
            layer.labelIds[i] = id;
        }

        JST_CHECK(window->build(layer.text, cfg));
        JST_CHECK(window->bind(layer.text));
        return Result::SUCCESS;
    };

    JST_CHECK(buildLabelLayer(Impl::LabelCountries, "default_mono_bold"));
    JST_CHECK(buildLabelLayer(Impl::LabelCapitals, "default_mono_bold"));
    JST_CHECK(buildLabelLayer(Impl::LabelCities, "default_mono"));
    JST_CHECK(buildLabelLayer(Impl::LabelStates, "default_mono_bold"));
    JST_CHECK(buildLabelLayer(Impl::LabelRegions, "default_body_italic"));
    JST_CHECK(buildLabelLayer(Impl::LabelMarine, "default_body_italic"));
    JST_CHECK(buildLabelLayer(Impl::LabelAirports, "default_mono"));
    JST_CHECK(buildLabelLayer(Impl::LabelPhysical, "default_body_italic"));

    // Build a compact one-character element pool for curved range labels.
    {
        U64 glyphCapacity = 0;
        for (const auto& label :
             pimpl->labelLayers[Impl::LabelRegions].labels) {
            if (!label.path.empty()) {
                glyphCapacity += label.name.size();
            }
        }
        glyphCapacity += Impl::ReferenceLabelGlyphs;
        glyphCapacity = std::min(glyphCapacity, Impl::PathLabelMaxGlyphs);
        if (glyphCapacity > 0) {
            std::shared_ptr<Render::Components::Font> font;
            if (window->hasFont("default_body_bold_italic")) {
                font = window->font("default_body_bold_italic");
            } else if (window->hasFont("default_mono_bold")) {
                font = window->font("default_mono_bold");
            }
            if (font) {
                Render::Components::Text::Config cfg;
                cfg.maxCharacters = 1;
                cfg.color = {0.78f, 0.66f, 0.48f, 0.88f};
                cfg.font = font;
                cfg.sharpness = 0.5f;
                pimpl->pathLabelIds.resize(glyphCapacity);
                for (U64 i = 0; i < glyphCapacity; ++i) {
                    const auto id = jst::fmt::format("p{:03d}", i);
                    cfg.elements[id] = {
                        .scale = 0.0f,
                        .alignment = {1, 1},
                    };
                    pimpl->pathLabelIds[i] = id;
                }
                JST_CHECK(window->build(pimpl->pathText, cfg));
                JST_CHECK(window->bind(pimpl->pathText));
            }
        }
    }

    // The same centered quad is shared by every point marker and the stars.
    {
        Render::Buffer::Config cfg;
        cfg.buffer = const_cast<F32*>(MarkerQuadVertices);
        cfg.elementByteSize = sizeof(F32);
        cfg.size = 12;
        cfg.target = Render::Buffer::Target::VERTEX;
        JST_CHECK(window->build(pimpl->markerQuadBuffer, cfg));
    }

    auto buildPointMarkers = [&](U64 markerCapacity,
                                 PointMarkerLayer& markers,
                                 const char* shaderName,
                                 F32 colorR, F32 colorG, F32 colorB) -> Result {
        if (markerCapacity == 0) return Result::SUCCESS;
        markers.instances.resize(markerCapacity * 4, 0.0f);
        markers.gpuUniforms.colorR = colorR;
        markers.gpuUniforms.colorG = colorG;
        markers.gpuUniforms.colorB = colorB;
        {
            Render::Buffer::Config cfg;
            cfg.buffer = markers.instances.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = markers.instances.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            cfg.enableZeroCopy = false;
            JST_CHECK(window->build(markers.instanceBuffer, cfg));
        }
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &markers.gpuUniforms;
            cfg.elementByteSize = sizeof(GpuUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(markers.uniformBuffer, cfg));
        }
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {{pimpl->markerQuadBuffer, 2}};
            cfg.instances = {{markers.instanceBuffer, 4}};
            JST_CHECK(window->build(markers.vertex, cfg));
        }
        {
            Render::Draw::Config cfg;
            cfg.buffer = markers.vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            cfg.numberOfInstances = 0;
            JST_CHECK(window->build(markers.draw, cfg));
        }
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage[shaderName];
            cfg.draws = {markers.draw};
            cfg.buffers = {{markers.uniformBuffer,
                            Render::Program::Target::VERTEX |
                            Render::Program::Target::FRAGMENT}};
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(markers.program, cfg));
        }
        return Result::SUCCESS;
    };

    U64 worldCityCount = 0;
    for (const U64 layerIndex : {Impl::LabelCapitals, Impl::LabelCities}) {
        for (const auto& label : pimpl->labelLayers[layerIndex].labels) {
            if (label.flags & 2) ++worldCityCount;
        }
    }
    JST_CHECK(buildPointMarkers(worldCityCount,
                                pimpl->worldCityMarkers,
                                "citymarker", 1.0f, 0.82f, 0.34f));
    const U64 cityMarkerCapacity =
        pimpl->labelLayers[Impl::LabelCities].poolSize;
    JST_CHECK(buildPointMarkers(cityMarkerCapacity,
                                pimpl->cityDotMarkers,
                                "citydot", 0.92f, 0.94f, 0.96f));
    JST_CHECK(buildPointMarkers(
        pimpl->labelLayers[Impl::LabelCapitals].poolSize,
        pimpl->capitalDotMarkers,
        "citydot", 1.0f, 0.82f, 0.34f));
    JST_CHECK(buildPointMarkers(
        pimpl->labelLayers[Impl::LabelAirports].labels.size(),
        pimpl->airportMarkers,
        "airport", 0.95f, 0.78f, 0.40f));

    // Build the background starfield: a deterministic set of stars on the
    // celestial sphere, rendered behind the globe with a skybox projection.
    {
        constexpr U64 kStarCount = 2500;
        const F32 goldenAngle = 2.39996323f;
        pimpl->stars.instances.reserve(kStarCount * 4);
        auto hash = [](U32 n) -> F32 {
            const F32 s = std::sin(static_cast<F32>(n) * 127.1f + 311.7f) *
                          43758.5453f;
            return s - std::floor(s);
        };
        for (U64 i = 0; i < kStarCount; ++i) {
            // Uniform distribution on a sphere via the golden-section spiral.
            const F32 y = 1.0f -
                (2.0f * static_cast<F32>(i) + 1.0f) /
                static_cast<F32>(kStarCount);
            const F32 r = std::sqrt(std::max(0.0f, 1.0f - y * y));
            const F32 theta = static_cast<F32>(i) * goldenAngle;
            // Brightness: power-law skew so most stars are dim, a few bright.
            const F32 b = 0.15f + 0.85f * std::pow(hash(static_cast<U32>(i) + 1u), 3.0f);
            pimpl->stars.instances.push_back(r * std::cos(theta));
            pimpl->stars.instances.push_back(y);
            pimpl->stars.instances.push_back(r * std::sin(theta));
            pimpl->stars.instances.push_back(b);
        }
        pimpl->stars.count = kStarCount;

        {
            Render::Buffer::Config cfg;
            cfg.buffer = pimpl->stars.instances.data();
            cfg.elementByteSize = sizeof(F32);
            cfg.size = pimpl->stars.instances.size();
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(pimpl->stars.instanceBuffer, cfg));
        }
        pimpl->stars.uniforms.skyViewProjection = pimpl->skyViewProjection;
        pimpl->stars.uniforms.viewportWidth = pimpl->uniforms.viewportWidth;
        pimpl->stars.uniforms.viewportHeight = pimpl->uniforms.viewportHeight;
        pimpl->stars.uniforms.surfaceScale = pimpl->uniforms.surfaceScale;
        {
            Render::Buffer::Config cfg;
            cfg.buffer = &pimpl->stars.uniforms;
            cfg.elementByteSize = sizeof(StarsUniforms);
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(pimpl->stars.uniformBuffer, cfg));
        }
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {{pimpl->markerQuadBuffer, 2}};
            cfg.instances = {{pimpl->stars.instanceBuffer, 4}};
            JST_CHECK(window->build(pimpl->stars.vertex, cfg));
        }
        {
            Render::Draw::Config cfg;
            cfg.buffer = pimpl->stars.vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            cfg.numberOfInstances = pimpl->stars.count;
            JST_CHECK(window->build(pimpl->stars.draw, cfg));
        }
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage["stars"];
            cfg.draws = {pimpl->stars.draw};
            cfg.buffers = {{pimpl->stars.uniformBuffer,
                            Render::Program::Target::VERTEX |
                            Render::Program::Target::FRAGMENT}};
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(pimpl->stars.program, cfg));
        }
    }

    return Result::SUCCESS;
}

Result GeoMapBaseLayer::destroy(Window* window) {
    for (auto& layer : pimpl->labelLayers) {
        if (layer.text) {
            JST_CHECK(window->unbind(layer.text));
        }
    }
    if (pimpl->pathText) {
        JST_CHECK(window->unbind(pimpl->pathText));
    }
    pimpl = std::make_unique<Impl>();
    return Result::SUCCESS;
}

Result GeoMapBaseLayer::surface(Render::Surface::Config& config) {
    // Background starfield: drawn first so the globe overpaints it where the
    // surface is covered, leaving stars visible only in the space around the
    // globe (no depth buffer — paint order alone decides occlusion).
    if (pimpl->stars.program) {
        config.programs.push_back(pimpl->stars.program);
    }

    // Base ocean sphere: underpaints every fill so mesh slivers near the
    // limb never reveal the starfield through the globe.
    if (pimpl->waterSphere.indexCount > 0) {
        config.programs.push_back(pimpl->waterSphere.program);
    }

    // Bathymetry (single merged draw call).
    if (pimpl->bathymetry.indexCount > 0) {
        config.programs.push_back(pimpl->bathymetry.program);
    }

    // Landcover: land + urban + lakes (single merged draw call).
    if (pimpl->landcover.indexCount > 0) {
        config.programs.push_back(pimpl->landcover.program);
    }

    // Geographic reference lines render on top of landcover.
    for (auto& g : pimpl->geo) {
        if (g.instanceCount > 0) {
            config.programs.push_back(g.program);
        }
    }

    for (auto it = pimpl->rivers.rbegin(); it != pimpl->rivers.rend(); ++it) {
        if (it->program) {
            config.programs.push_back(it->program);
        }
    }

    if (pimpl->stateBorders.instanceCount > 0) {
        config.programs.push_back(pimpl->stateBorders.program);
    }
    if (pimpl->countryBorders.program) {
        config.programs.push_back(pimpl->countryBorders.program);
    }
    if (pimpl->coastlines.instanceCount > 0) {
        config.programs.push_back(pimpl->coastlines.program);
    }
    // Draw generic references first and politically significant classes last.
    for (auto it = pimpl->disputed.rbegin();
         it != pimpl->disputed.rend(); ++it) {
        auto& category = *it;
        if (category.program) {
            config.programs.push_back(category.program);
        }
    }
    // Atmospheric rim and outer halo haze the fills and map lines near the
    // limb alike, but stay under markers and labels.
    if (pimpl->atmosphere.program) {
        config.programs.push_back(pimpl->atmosphere.program);
    }
    if (pimpl->airportMarkers.program) {
        config.programs.push_back(pimpl->airportMarkers.program);
    }
    if (pimpl->cityDotMarkers.program) {
        config.programs.push_back(pimpl->cityDotMarkers.program);
    }
    if (pimpl->capitalDotMarkers.program) {
        config.programs.push_back(pimpl->capitalDotMarkers.program);
    }
    if (pimpl->worldCityMarkers.program) {
        config.programs.push_back(pimpl->worldCityMarkers.program);
    }

    // Lower-priority labels draw first; collision winners draw last.
    std::array<U64, Impl::NumLabelLayers> order{};
    for (U64 i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](U64 a, U64 b) {
        return pimpl->labelLayers[a].priority >
               pimpl->labelLayers[b].priority;
    });
    for (U64 idx : order) {
        auto& layer = pimpl->labelLayers[idx];
        if (layer.text) {
            JST_CHECK(layer.text->surface(config));
        }
        if (idx == Impl::LabelRegions && pimpl->pathText) {
            JST_CHECK(pimpl->pathText->surface(config));
        }
    }

    return Result::SUCCESS;
}

Result GeoMapBaseLayer::present(const MapContext& context) {
    const bool updateView = pimpl->firstFrame || pimpl->uniforms != context.view;
    pimpl->uniforms = context.view;
    pimpl->cameraUniforms = context.camera;
    pimpl->skyViewProjection = context.skyViewProjection;
    if (updateView) {
        auto copyCamera = [&](GpuUniforms& gpu) {
            CopyCameraFields(gpu, pimpl->cameraUniforms);
        };

        auto updateLine = [&](LineCategory& line, bool enabled, F32 opacity) -> Result {
            if (!line.program) return Result::SUCCESS;
            copyCamera(line.gpuUniforms);
            line.gpuUniforms.lineOpacity = opacity;
            JST_CHECK(line.uniformBuffer->update());
            return line.draw->updateInstanceCount(enabled ? line.instanceCount : 0);
        };
        JST_CHECK(updateLine(pimpl->coastlines, pimpl->tuning.coastlines,
                              pimpl->tuning.coastlineOpacity));
        JST_CHECK(updateLine(pimpl->countryBorders, pimpl->tuning.countryBorders,
                              pimpl->tuning.countryBorderOpacity));
        JST_CHECK(updateLine(pimpl->stateBorders, pimpl->tuning.stateBorders,
            GeoMapLabels::VisibilityFade(pimpl->uniforms.detailZoom,
                pimpl->tuning.stateLineFull,
                pimpl->tuning.stateLineFull - pimpl->tuning.stateLineFadeStart) *
            pimpl->tuning.stateLineOpacity));

        for (auto& river : pimpl->rivers) {
            if (!river.program) continue;
            copyCamera(river.gpuUniforms);
            river.gpuUniforms.lineOpacity = GeoMapLabels::VisibilityFade(
                pimpl->uniforms.detailZoom,
                river.fullVisibilityZoom +
                    pimpl->tuning.riverZoomOffset,
                pimpl->tuning.riverFadeRange);
            river.uniformBuffer->update();
            JST_CHECK(river.draw->updateInstanceCount(
                pimpl->tuning.rivers &&
                river.gpuUniforms.lineOpacity > 0.0f
                    ? river.instanceCount
                    : 0));
        }

        for (auto& g : pimpl->geo) {
            if (g.instanceCount > 0) {
                copyCamera(g.gpuUniforms);
                g.gpuUniforms.lineOpacity =
                    GeoMapLabels::VisibilityFadeOut(
                        pimpl->uniforms.detailZoom,
                        pimpl->tuning.referenceMaxZoom[
                            static_cast<U64>(&g - pimpl->geo.data())],
                        pimpl->tuning.referenceFadeRange);
                g.gpuUniforms.dashPhase = g.baseDashPhase;
                if (g.gpuUniforms.lineStyle > 0.5f) {
                    GeoMapLines::UpdateDashRanges(g.vertices, context, g.dashRanges);
                    JST_CHECK(g.dashBuffer->update());
                }
                g.uniformBuffer->update();
                JST_CHECK(g.draw->updateInstanceCount(
                    pimpl->tuning.references ? g.instanceCount : 0));
            }
        }

        for (auto& category : pimpl->disputed) {
            if (!category.program) continue;
            U64 visibleCount = 0;
            for (U64 i = 0; i < category.minZooms.size(); ++i) {
                if (pimpl->uniforms.detailZoom <
                    category.minZooms[i] +
                        pimpl->tuning.disputedZoomOffset) {
                    continue;
                }
                std::copy_n(category.vertices.data() + i * 4, 4,
                            category.visibleVertices.data() +
                                visibleCount * 4);
                ++visibleCount;
            }
            category.instanceCount = visibleCount;
            copyCamera(category.gpuUniforms);
            category.gpuUniforms.dashPhase = category.baseDashPhase;
            category.instanceBuffer->update();
            category.uniformBuffer->update();
            JST_CHECK(category.draw->updateInstanceCount(
                pimpl->tuning.disputedBorders ? visibleCount : 0));
        }

        // Update merged fill layer uniforms. The globe is a single copy — the
        // sphere handles antimeridian wrapping natively, so no world offsets.
        auto updateFillUniforms = [&](MergedFillLayer& layer,
                                      bool enabled) -> Result {
            if (layer.indexCount > 0) {
                copyCamera(layer.gpuUniforms);
                layer.gpuUniforms.lineWidth = 0.0f;
                layer.uniformBuffer->update();
                JST_CHECK(layer.draw->updateInstanceCount(enabled ? 1 : 0));
            }
            return Result::SUCCESS;
        };

        JST_CHECK(updateFillUniforms(pimpl->waterSphere,
                                     pimpl->tuning.waterSphere));
        JST_CHECK(updateFillUniforms(pimpl->bathymetry,
                                     pimpl->tuning.bathymetry));
        JST_CHECK(updateFillUniforms(pimpl->landcover,
                                     pimpl->tuning.landcover));
        if (pimpl->atmosphere.program) {
            copyCamera(pimpl->atmosphere.gpuUniforms);
            pimpl->atmosphere.gpuUniforms.lineWidth =
                pimpl->tuning.atmosphereRadius;
            pimpl->atmosphere.gpuUniforms.colorR =
                pimpl->tuning.atmosphereColor[0];
            pimpl->atmosphere.gpuUniforms.colorG =
                pimpl->tuning.atmosphereColor[1];
            pimpl->atmosphere.gpuUniforms.colorB =
                pimpl->tuning.atmosphereColor[2];
            pimpl->atmosphere.gpuUniforms.dashScale =
                pimpl->tuning.atmosphereInnerStrength;
            pimpl->atmosphere.gpuUniforms.dashPhase =
                pimpl->tuning.atmosphereOuterStrength;
            pimpl->atmosphere.gpuUniforms.outlineStrength =
                pimpl->tuning.atmosphereOutlineStrength;
            pimpl->atmosphere.uniformBuffer->update();
            JST_CHECK(pimpl->atmosphere.draw->updateInstanceCount(
                pimpl->tuning.atmosphere ? 1 : 0));
        }

        // Update the starfield uniforms so the celestial sphere tracks the
        // 3D globe camera (stars rotate opposite to panning).
        if (pimpl->stars.program) {
            pimpl->stars.uniforms.skyViewProjection = pimpl->skyViewProjection;
            pimpl->stars.uniforms.viewportWidth = pimpl->uniforms.viewportWidth;
            pimpl->stars.uniforms.viewportHeight = pimpl->uniforms.viewportHeight;
            pimpl->stars.uniforms.surfaceScale = pimpl->uniforms.surfaceScale;
            pimpl->stars.uniformBuffer->update();
        }

        // Update labels using the documented semantic zoom policy above,
        // falling back to Natural Earth thresholds for untiered features.
        const auto pixelSize = context.pixelSize();
        const F32 detailZoom = pimpl->uniforms.detailZoom;

        // Project a geographic position through the 3D globe camera to
        // normalized device coordinates. Returns false for points on the far
        // side of the globe (beyond the horizon), which hides their labels.
        const glm::vec3 tn(pimpl->cameraUniforms.targetNormal);
        const F32 thr = pimpl->cameraUniforms.targetNormal.w;
        auto projectLabel = [&](F32 lon, F32 lat, F32& ndcX, F32& ndcY) -> bool {
            return context.projectLonLat(lon, lat, ndcX, ndcY);
        };

        auto fullVisibilityZoom = [&](const LabelInfo& lbl,
                                      const Impl::LabelLayer& layer) {
            return GeoMapLabels::FullVisibilityZoom(lbl, pimpl->tuning,
                layer.minZoomFloor, layer.visibilityOffset);
        };
        auto maxVisibilityZoom = [&](const LabelInfo& lbl,
                                     const Impl::LabelLayer& layer) {
            if (!lbl.path.empty()) {
                return lbl.maxZoom > 0.0f
                    ? lbl.maxZoom
                    : pimpl->tuning.pathLabelDefaultHidden;
            }
            if (lbl.maxZoom <= 0.0f) return 0.0f;
            return std::max(lbl.maxZoom, layer.maxZoomFloor);
        };
        auto fadeRange = [](const LabelInfo& lbl,
                            const Impl::LabelLayer& layer) {
            if (lbl.kind == LabelKind::Country) {
                return GeoMapLabels::CountryFadeRange(lbl.scalerank);
            }
            if (!lbl.path.empty()) {
                return 1.0f;
            }
            return layer.fadeRange;
        };
        auto labelFade = [&](const LabelInfo& lbl, const Impl::LabelLayer& layer,
                              const LabelTuning& tuning) {
            return GeoMapLabels::Visibility(detailZoom,
                fullVisibilityZoom(lbl, layer) + tuning.zoomOffset,
                maxVisibilityZoom(lbl, layer), fadeRange(lbl, layer) * tuning.fadeScale);
        };

        // Screen-space collision list. Every rendered label and marker box
        // is kept with its alpha so later candidates cross-fade against it.
        struct AABB { F32 x0, y0, x1, y1; };
        struct RenderedBox { AABB box; F32 alpha; };
        std::vector<RenderedBox> rendered;
        U64 worldCityMarkerCount = 0;
        U64 capitalDotMarkerCount = 0;
        U64 cityDotMarkerCount = 0;
        U64 airportMarkerCount = 0;
        U64 totalPoolSize = 0;
        for (const auto& layer : pimpl->labelLayers) {
            totalPoolSize += layer.poolSize;
        }
        rendered.reserve(totalPoolSize * 2);

        // Process layers by ascending priority.
        std::array<U64, Impl::NumLabelLayers> order{};
        for (U64 i = 0; i < order.size(); ++i) order[i] = i;
        std::sort(order.begin(), order.end(),
                  [&](U64 a, U64 b) {
                      return pimpl->labelLayers[a].priority <
                             pimpl->labelLayers[b].priority;
                  });

        // Highest alpha among already rendered boxes overlapping `b`. A
        // candidate scales its own alpha by one minus this value, so a more
        // important label fading in gradually dims what it covers and a fully
        // opaque one hides it outright.
        auto occlusion = [&](const AABB& b) -> F32 {
            F32 result = 0.0f;
            for (const auto& r : rendered) {
                const auto& p = r.box;
                if (b.x0 < p.x1 && b.x1 > p.x0 &&
                    b.y0 < p.y1 && b.y1 > p.y0) {
                    result = std::max(result, r.alpha);
                }
            }
            return result;
        };
        constexpr F32 kOpaqueAlpha = 0.999f;

        struct PathGlyph {
            char character;
            F32 x;
            F32 y;
            F32 angleDeg;
            AABB box;
        };
        const F32 pathLineHeight = pimpl->pathText
            ? static_cast<F32>(
                  pimpl->pathText->getConfig().font->lineHeight())
            : 0.0f;
        U64 pathGlyphSlot = 0;
        if (pimpl->pathText) {
            JST_CHECK(pimpl->pathText->updatePixelSize(pixelSize));
        }

        // Lay one label along a geographic path with glyphs fixed in screen
        // pixels. Glyphs center on the path midpoint and sit `offsetPixels`
        // to the path's left-hand side, so a west-to-east path carries its
        // label just north of the line it names.
        auto layoutPathLabel = [&](const std::string& name,
                                   const std::vector<LabelPathPoint>& path,
                                   F32 scale,
                                   F32 tracking,
                                   F32 offsetPixels,
                                   ColorRGBA<F32> color,
                                   F32 fade) -> Result {
            if (!pimpl->pathText || path.size() < 2 || name.empty()) {
                return Result::SUCCESS;
            }
            if (pathGlyphSlot + name.size() > pimpl->pathLabelIds.size()) {
                return Result::SUCCESS;
            }

            std::vector<Extent2D<F32>> pathPixels;
            std::vector<F32> cumulative;
            pathPixels.reserve(path.size());
            cumulative.reserve(path.size());
            for (const auto& point : path) {
                F32 ndcX, ndcY;
                // A path crossing the globe's horizon can't be laid out
                // cleanly in screen space, so hide the whole label.
                if (!projectLabel(point.lon, point.lat, ndcX, ndcY)) {
                    return Result::SUCCESS;
                }
                pathPixels.push_back({ndcX / pixelSize.x,
                                      ndcY / pixelSize.y});
                if (pathPixels.size() == 1) {
                    cumulative.push_back(0.0f);
                } else {
                    const auto& a = pathPixels[pathPixels.size() - 2];
                    const auto& b = pathPixels.back();
                    cumulative.push_back(
                        cumulative.back() + std::hypot(b.x - a.x,
                                                       b.y - a.y));
                }
            }

            const F32 pathLength = cumulative.back();
            const auto advances = pimpl->pathText->advances(name);
            F32 naturalWidth = 0.0f;
            for (const F32 advance : advances) {
                naturalWidth += advance * scale;
            }
            const F32 layoutWidth = naturalWidth +
                tracking * static_cast<F32>(name.size() - 1);
            const F32 labelSpan = GeoMapLabels::FixedPathLabelSpan(
                layoutWidth, pathLength);
            if (labelSpan <= 0.0f) return Result::SUCCESS;
            const F32 pathNormalization =
                GeoMapLabels::FixedPathNormalization(labelSpan, pathLength);

            struct PathSample {
                F32 x;
                F32 y;
                F32 angle;
            };
            auto samplePath = [&](F32 distance) {
                const auto segment = std::upper_bound(
                    cumulative.begin(), cumulative.end(), distance);
                const U64 endIndex = std::clamp<U64>(
                    static_cast<U64>(segment - cumulative.begin()),
                    1, cumulative.size() - 1);
                const U64 startIndex = endIndex - 1;
                const F32 segmentLength = cumulative[endIndex] -
                                          cumulative[startIndex];
                const F32 amount = segmentLength > 1.0e-5f
                    ? (distance - cumulative[startIndex]) / segmentLength
                    : 0.0f;
                const auto& a = pathPixels[startIndex];
                const auto& b = pathPixels[endIndex];
                return PathSample{
                    std::lerp(a.x, b.x, amount),
                    std::lerp(a.y, b.y, amount),
                    std::atan2(b.y - a.y, b.x - a.x),
                };
            };
            const PathSample pathCenter = samplePath(pathLength * 0.5f);

            std::vector<PathGlyph> glyphs;
            glyphs.reserve(name.size());
            F32 cursor = 0.0f;
            F32 pathOcclusion = 0.0f;
            for (U64 i = 0; i < name.size(); ++i) {
                const F32 glyphWidth = advances[i] * scale;
                const F32 fraction = (cursor + glyphWidth * 0.5f) / layoutWidth;
                const PathSample sample = samplePath(pathLength * fraction);
                const F32 angle = sample.angle;
                const F32 pixelX = pathCenter.x +
                    (sample.x - pathCenter.x) * pathNormalization -
                    std::sin(angle) * offsetPixels;
                const F32 pixelY = pathCenter.y +
                    (sample.y - pathCenter.y) * pathNormalization +
                    std::cos(angle) * offsetPixels;
                const F32 halfWidth = glyphWidth * 0.5f;
                const F32 halfHeight = pathLineHeight * scale * 0.5f;
                const F32 boxHalfWidth =
                    std::abs(std::cos(angle)) * halfWidth +
                    std::abs(std::sin(angle)) * halfHeight;
                const F32 boxHalfHeight =
                    std::abs(std::sin(angle)) * halfWidth +
                    std::abs(std::cos(angle)) * halfHeight;
                const F32 ndcX = pixelX * pixelSize.x;
                const F32 ndcY = pixelY * pixelSize.y;
                const AABB box = {
                    ndcX - boxHalfWidth * pixelSize.x,
                    ndcY - boxHalfHeight * pixelSize.y,
                    ndcX + boxHalfWidth * pixelSize.x,
                    ndcY + boxHalfHeight * pixelSize.y,
                };
                if (box.x0 < -1.05f || box.x1 > 1.05f ||
                    box.y0 < -1.05f || box.y1 > 1.05f) {
                    return Result::SUCCESS;
                }
                pathOcclusion = std::max(pathOcclusion, occlusion(box));
                if (pathOcclusion >= kOpaqueAlpha) {
                    return Result::SUCCESS;
                }
                glyphs.push_back({
                    name[i], ndcX, ndcY, angle * 180.0f / kPi, box,
                });
                cursor += glyphWidth + tracking;
            }
            fade *= 1.0f - pathOcclusion;
            if (fade <= 0.01f) return Result::SUCCESS;
            color.a *= fade;

            for (const auto& glyph : glyphs) {
                const auto& id = pimpl->pathLabelIds[pathGlyphSlot++];
                auto element = pimpl->pathText->get(id);
                element.position = {glyph.x, glyph.y};
                element.scale = scale;
                element.rotationDeg = glyph.angleDeg;
                element.color = color;
                const std::string fill(1, glyph.character);
                if (element.fill != fill) element.fill = fill;
                JST_CHECK(pimpl->pathText->update(id, element));
                rendered.push_back({glyph.box, fade});
            }
            return Result::SUCCESS;
        };

        auto updatePathLabels = [&](const Impl::LabelLayer& layer) -> Result {
            const auto& pathTuning =
                pimpl->tuning.labels[Impl::LabelRegions];
            if (!pimpl->pathText || !pathTuning.enabled) {
                return Result::SUCCESS;
            }
            for (const auto& lbl : layer.labels) {
                if (lbl.path.size() < 2 || lbl.name.empty()) continue;

                const F32 fade = labelFade(lbl, layer, pathTuning);
                if (fade <= 0.01f) continue;

                JST_CHECK(layoutPathLabel(lbl.name, lbl.path, 1.0f, 10.0f,
                                          0.0f,
                                          {0.78f, 0.66f, 0.48f, 0.88f},
                                          fade));
            }
            return Result::SUCCESS;
        };

        // Reference line names ride the line itself. Parallels carry four
        // labels fixed to the globe at the Greenwich meridian and every 90
        // degrees from it; the date line has no natural anchor along its
        // length, so its label follows the view center's latitude. The
        // sampled arc is sized from the on-screen length of one degree so
        // the glyphs follow the line's curvature closely.
        auto updateReferenceLabels = [&]() -> Result {
            if (!pimpl->pathText || !pimpl->tuning.references) {
                return Result::SUCCESS;
            }
            struct ReferenceLabel {
                U64 group;
                const char* name;
                F32 coordinate;
                bool meridian;
            };
            static constexpr ReferenceLabel references[] = {
                {Impl::GeoEquator, "EQUATOR", 0.0f, false},
                {Impl::GeoTropics, "TROPIC OF CANCER", 23.43693f, false},
                {Impl::GeoTropics, "TROPIC OF CAPRICORN", -23.43693f, false},
                {Impl::GeoPolar, "ARCTIC CIRCLE", 66.56307f, false},
                {Impl::GeoPolar, "ANTARCTIC CIRCLE", -66.56307f, false},
                {Impl::GeoDateLine, "INTERNATIONAL DATE LINE", 180.0f, true},
            };
            constexpr F32 scale = 0.46f;
            constexpr F32 tracking = 3.0f;
            constexpr U64 sampleCount = 9;
            constexpr F32 maxLatitude = 85.0f;
            static constexpr F32 parallelAnchors[] = {
                0.0f, 90.0f, 180.0f, -90.0f,
            };
            const F32 centerLat = std::clamp(pimpl->uniforms.centerLat,
                                             -maxLatitude, maxLatitude);
            const ColorRGBA<F32> color = {0.58f, 0.66f, 0.72f, 0.80f};

            auto placeReference = [&](const ReferenceLabel& ref,
                                      F32 anchorLon,
                                      F32 anchorLat,
                                      F32 fade) -> Result {
                const std::string name(ref.name);
                const auto advances = pimpl->pathText->advances(name);
                F32 layoutWidth =
                    tracking * static_cast<F32>(name.size() - 1);
                for (const F32 advance : advances) {
                    layoutWidth += advance * scale;
                }

                const F32 probeLon = ref.meridian
                    ? anchorLon
                    : WrapLongitude(anchorLon + 1.0f);
                const F32 probeLat = ref.meridian
                    ? (anchorLat > maxLatitude - 1.0f
                           ? anchorLat - 1.0f
                           : anchorLat + 1.0f)
                    : anchorLat;
                F32 anchorX, anchorY, probeX, probeY;
                if (!projectLabel(anchorLon, anchorLat, anchorX, anchorY) ||
                    !projectLabel(probeLon, probeLat, probeX, probeY)) {
                    return Result::SUCCESS;
                }
                const F32 pixelsPerDegree = std::hypot(
                    (probeX - anchorX) / pixelSize.x,
                    (probeY - anchorY) / pixelSize.y);
                if (pixelsPerDegree <= 1.0e-3f) return Result::SUCCESS;
                const F32 halfSpan = std::clamp(
                    0.6f * layoutWidth / pixelsPerDegree, 0.05f, 60.0f);

                std::vector<LabelPathPoint> path(sampleCount);
                for (U64 i = 0; i < sampleCount; ++i) {
                    const F32 t = static_cast<F32>(i) /
                        static_cast<F32>(sampleCount - 1) * 2.0f - 1.0f;
                    if (ref.meridian) {
                        path[i].lon = anchorLon;
                        path[i].lat = std::clamp(anchorLat + t * halfSpan,
                                                 -maxLatitude, maxLatitude);
                    } else {
                        path[i].lon = WrapLongitude(anchorLon + t * halfSpan);
                        path[i].lat = anchorLat;
                    }
                }
                return layoutPathLabel(
                    name, path, scale, tracking,
                    pathLineHeight * scale * 0.5f + 2.0f, color, fade);
            };

            for (const auto& ref : references) {
                const auto& line = pimpl->geo[ref.group];
                const F32 fade = line.instanceCount > 0
                    ? line.gpuUniforms.lineOpacity
                    : 0.0f;
                if (fade <= 0.01f) continue;
                if (ref.meridian) {
                    JST_CHECK(placeReference(ref, ref.coordinate, centerLat,
                                             fade));
                    continue;
                }
                for (const F32 anchorLon : parallelAnchors) {
                    JST_CHECK(placeReference(ref, anchorLon, ref.coordinate,
                                             fade));
                }
            }
            return Result::SUCCESS;
        };

        for (U64 oi = 0; oi < order.size(); ++oi) {
            const U64 idx = order[oi];
            auto& layer = pimpl->labelLayers[idx];
            if (idx == Impl::LabelRegions) {
                JST_CHECK(updatePathLabels(layer));
            }
            if (!layer.text || layer.labels.empty()) {
                continue;
            }
            JST_CHECK(layer.text->updatePixelSize(pixelSize));
            const F32 padX = pixelSize.x * layer.collisionPaddingPixels;
            const F32 padY = pixelSize.y * layer.collisionPaddingPixels;

            U64 slot = 0;
            const auto& labelTuning = pimpl->tuning.labels[idx];
            const U64 slotLimit = labelTuning.enabled ? layer.poolSize : 0;
            for (const auto& lbl : layer.labels) {
                if (slot >= slotLimit) {
                    break;
                }
                if (idx == Impl::LabelRegions && !lbl.path.empty()) {
                    continue;
                }
                const bool isWorldCity =
                    (idx == Impl::LabelCapitals ||
                     idx == Impl::LabelCities) &&
                    (lbl.flags & 2);
                const bool isCityLabel =
                    idx == Impl::LabelCapitals || idx == Impl::LabelCities;
                const bool cityMarkerEnabled = isWorldCity
                    ? pimpl->tuning.worldCityMarkers
                    : idx == Impl::LabelCapitals
                        ? pimpl->tuning.capitalMarkers
                        : pimpl->tuning.cityMarkers;
                const bool airportMarkerEnabled =
                    pimpl->tuning.airportMarkers;
                if (idx == Impl::LabelAirports && !(lbl.flags & 4)) {
                    continue;
                }

                F32 fade = labelFade(lbl, layer, labelTuning);
                if (fade <= 0.01f) continue;

                F32 ndcX, ndcY;
                if (!projectLabel(lbl.lon, lbl.lat, ndcX, ndcY)) {
                    continue;  // on the far side of the globe
                }
                if (std::abs(ndcX) > 1.2f || std::abs(ndcY) > 1.2f) {
                    continue;
                }
                if (layer.limbFadeStart > 0.0f) {
                    const F32 facing = glm::dot(
                        LonLatToSphere(lbl.lon, lbl.lat), tn);
                    const F32 limb = (facing - thr) /
                                     std::max(1.0f - thr, 1.0e-4f);
                    fade *= GeoMapLabels::VisibilityFade(
                        limb, layer.limbFadeStart + layer.limbFadeRange,
                        layer.limbFadeRange);
                    if (fade <= 0.01f) continue;
                }

                F32 occluded = 0.0f;
                AABB cityMarkerBox{};
                F32 cityMarkerRadius = 0.0f;
                if (isCityLabel && cityMarkerEnabled) {
                    cityMarkerRadius = isWorldCity
                        ? (lbl.flags & 8 ? 7.0f : 6.0f) *
                          (1.0f + std::clamp(detailZoom, 0.0f, 8.0f) * 0.06f)
                        : 5.5f *
                          (1.0f + std::clamp(detailZoom, 0.0f, 8.0f) * 0.025f);
                    const F32 markerHalfWidth =
                        pixelSize.x * (cityMarkerRadius + 2.0f);
                    const F32 markerHalfHeight =
                        pixelSize.y * (cityMarkerRadius + 2.0f);
                    cityMarkerBox = {
                        ndcX - markerHalfWidth, ndcY - markerHalfHeight,
                        ndcX + markerHalfWidth, ndcY + markerHalfHeight,
                    };
                    if (cityMarkerBox.x0 < -1.05f ||
                        cityMarkerBox.x1 > 1.05f ||
                        cityMarkerBox.y0 < -1.05f ||
                        cityMarkerBox.y1 > 1.05f) {
                        continue;
                    }
                    occluded = std::max(occluded, occlusion(cityMarkerBox));
                    if (occluded >= kOpaqueAlpha) continue;
                }

                AABB airportMarkerBox{};
                F32 airportMarkerRadius = 0.0f;
                if (idx == Impl::LabelAirports && airportMarkerEnabled) {
                    airportMarkerRadius = lbl.flags & 1 ? 8.0f
                        : lbl.flags & 2 ? 7.0f : 6.0f;
                    const F32 markerHalfWidth =
                        pixelSize.x * (airportMarkerRadius + 2.0f);
                    const F32 markerHalfHeight =
                        pixelSize.y * (airportMarkerRadius + 2.0f);
                    airportMarkerBox = {
                        ndcX - markerHalfWidth, ndcY - markerHalfHeight,
                        ndcX + markerHalfWidth, ndcY + markerHalfHeight,
                    };
                    if (airportMarkerBox.x0 < -1.05f ||
                        airportMarkerBox.x1 > 1.05f ||
                        airportMarkerBox.y0 < -1.05f ||
                        airportMarkerBox.y1 > 1.05f) {
                        continue;
                    }
                    occluded = std::max(occluded,
                                        occlusion(airportMarkerBox));
                    if (occluded >= kOpaqueAlpha) continue;
                }

                F32 lblScale = layer.baseScale;
                ColorRGBA<F32> lblColor = {1.0f, 1.0f, 1.0f, 0.90f};
                bool upper = layer.uppercase;
                std::string fill = lbl.name;

                switch (idx) {
                    case Impl::LabelCountries: {
                        lblColor = {0.96f, 0.97f, 1.00f, 0.95f};
                        const F32 rankFactor = lbl.scalerank <= 2 ? 1.0f
                            : lbl.scalerank <= 4 ? 0.90f
                            : lbl.scalerank <= 6 ? 0.82f
                            : 0.76f;
                        const F32 zoomFactor = std::lerp(
                            0.68f, 1.0f,
                            std::clamp((detailZoom + 0.5f) / 2.5f,
                                       0.0f, 1.0f));
                        lblScale *= rankFactor * zoomFactor;
                        break;
                    }
                    case Impl::LabelCapitals: {
                        const F32 tier = lbl.flags & 8 ? 1.18f : 1.08f;
                        lblColor = lbl.flags & 8
                                       ? ColorRGBA<F32>{1.00f, 0.90f, 0.50f, 0.96f}
                                       : ColorRGBA<F32>{1.00f, 0.95f, 0.70f, 0.92f};
                        const F32 popFactor = 1.0f + std::clamp(
                            std::log10(static_cast<F32>(std::max<I32>(
                                1, lbl.population))) / 35.0f,
                            0.0f, 0.20f);
                        const F32 zoomFactor = 1.0f + std::clamp(
                            (detailZoom - 2.0f) * 0.03f, 0.0f, 0.20f);
                        lblScale *= tier * popFactor * zoomFactor;
                        break;
                    }
                    case Impl::LabelCities: {
                        F32 tier = 1.0f;
                        if (lbl.flags & 2) {
                            tier = 1.16f;
                            lblColor = {1.00f, 0.88f, 0.46f, 0.95f};
                        } else if (lbl.flags & 4) {
                            tier = 1.10f;
                            lblColor = {0.95f, 0.96f, 0.99f, 0.92f};
                        } else {
                            lblColor = {0.92f, 0.94f, 0.96f, 0.90f};
                        }
                        const F32 popFactor = 1.0f + std::clamp(
                            std::log10(static_cast<F32>(std::max<I32>(
                                1, lbl.population))) / 40.0f,
                            0.0f, 0.16f);
                        lblScale *= tier * popFactor;
                        if (lbl.flags & 2) {
                            lblScale *= 1.0f + std::clamp(
                                (detailZoom - 2.0f) * 0.03f,
                                0.0f, 0.20f);
                        }
                        break;
                    }
                    case Impl::LabelStates:
                        lblColor = {0.72f, 0.78f, 0.86f, 0.80f};
                        break;
                    case Impl::LabelRegions:
                        lblColor = {0.78f, 0.66f, 0.48f, 0.88f};
                        break;
                    case Impl::LabelMarine:
                        lblColor = {0.50f, 0.68f, 0.82f, 0.85f};
                        upper = true;
                        break;
                    case Impl::LabelAirports:
                        lblColor = {0.95f, 0.78f, 0.40f, 0.92f};
                        if (lbl.flags & 1) lblScale *= 1.15f;
                        break;
                    case Impl::LabelPhysical:
                        if (lbl.kind == LabelKind::Physical) {
                            lblColor = {0.74f, 0.62f, 0.45f, 0.85f};
                            lblScale *= 0.95f;
                        } else {
                            lblColor = {0.50f, 0.68f, 0.82f, 0.85f};
                        }
                        break;
                }

                if (upper) {
                    UppercaseAscii(fill);
                }

                F32 labelX = ndcX;
                F32 labelY = ndcY;
                if (isWorldCity) {
                    labelY += pixelSize.y * 28.0f;
                } else if (isCityLabel) {
                    labelY += pixelSize.y * (cityMarkerRadius + 17.0f);
                } else if (idx == Impl::LabelAirports) {
                    labelY += pixelSize.y * 22.0f;
                }
                const F32 w = layer.text->advance(fill) * pixelSize.x *
                              lblScale + padX * 2.0f;
                const U64 lineCount = static_cast<U64>(
                    std::count(fill.begin(), fill.end(), '\n')) + 1;
                const F32 h = static_cast<F32>(
                                  layer.text->getConfig().font->lineHeight()) *
                              static_cast<F32>(lineCount) *
                              pixelSize.y * lblScale + padY * 2.0f;
                const AABB box = {labelX - w * 0.5f, labelY - h * 0.5f,
                                  labelX + w * 0.5f, labelY + h * 0.5f};
                if (box.x0 < -1.05f || box.x1 > 1.05f ||
                    box.y0 < -1.05f || box.y1 > 1.05f) {
                    continue;
                }
                occluded = std::max(occluded, occlusion(box));
                if (occluded >= kOpaqueAlpha) continue;
                fade *= 1.0f - occluded;
                if (fade <= 0.01f) continue;
                lblColor.a *= fade;

                if (isCityLabel && cityMarkerEnabled) {
                    auto& markerLayer = isWorldCity
                        ? pimpl->worldCityMarkers
                        : idx == Impl::LabelCapitals
                            ? pimpl->capitalDotMarkers
                            : pimpl->cityDotMarkers;
                    U64& markerCount = isWorldCity
                        ? worldCityMarkerCount
                        : idx == Impl::LabelCapitals
                            ? capitalDotMarkerCount
                            : cityDotMarkerCount;
                    if (markerCount >= markerLayer.instances.size() / 4) {
                        continue;
                    }
                    const U64 base = markerCount++ * 4;
                    markerLayer.instances[base + 0] = lbl.lon;
                    markerLayer.instances[base + 1] = lbl.lat;
                    markerLayer.instances[base + 2] = cityMarkerRadius;
                    markerLayer.instances[base + 3] = fade;
                    rendered.push_back({cityMarkerBox, fade});
                }
                if (idx == Impl::LabelAirports && airportMarkerEnabled) {
                    auto& markerLayer = pimpl->airportMarkers;
                    if (airportMarkerCount >=
                        markerLayer.instances.size() / 4) {
                        continue;
                    }
                    const U64 base = airportMarkerCount++ * 4;
                    markerLayer.instances[base + 0] = lbl.lon;
                    markerLayer.instances[base + 1] = lbl.lat;
                    markerLayer.instances[base + 2] = airportMarkerRadius;
                    markerLayer.instances[base + 3] = fade;
                    rendered.push_back({airportMarkerBox, fade});
                }
                rendered.push_back({box, fade});

                const auto& id = layer.labelIds[slot];
                auto element = layer.text->get(id);
                element.position = {labelX, labelY};
                element.scale = lblScale;
                element.color = lblColor;
                if (element.fill != fill) {
                    element.fill = fill;
                }
                JST_CHECK(layer.text->update(id, element));

                ++slot;
            }

            for (U64 i = slot; i < layer.previousSlotCount; ++i) {
                const auto& id = layer.labelIds[i];
                auto element = layer.text->get(id);
                element.fill = "";
                element.scale = 0.0f;
                element.color = std::nullopt;
                JST_CHECK(layer.text->update(id, element));
            }
            layer.previousSlotCount = slot;
        }

        JST_CHECK(updateReferenceLabels());
        if (pimpl->pathText) {
            for (U64 i = pathGlyphSlot;
                 i < pimpl->previousPathGlyphCount; ++i) {
                const auto& id = pimpl->pathLabelIds[i];
                auto element = pimpl->pathText->get(id);
                element.fill = "";
                element.scale = 0.0f;
                element.color = std::nullopt;
                JST_CHECK(pimpl->pathText->update(id, element));
            }
            pimpl->previousPathGlyphCount = pathGlyphSlot;
        }

        auto uploadCityMarkers = [&](PointMarkerLayer& markers,
                                     U64 markerCount) -> Result {
            if (!markers.program) return Result::SUCCESS;
            CopyCameraFields(markers.gpuUniforms, pimpl->cameraUniforms);
            markers.instanceBuffer->update();
            markers.uniformBuffer->update();
            JST_CHECK(markers.draw->updateInstanceCount(markerCount));
            return Result::SUCCESS;
        };
        JST_CHECK(uploadCityMarkers(pimpl->worldCityMarkers,
                                    worldCityMarkerCount));
        JST_CHECK(uploadCityMarkers(pimpl->cityDotMarkers,
                                    cityDotMarkerCount));
        JST_CHECK(uploadCityMarkers(pimpl->capitalDotMarkers,
                                    capitalDotMarkerCount));
        JST_CHECK(uploadCityMarkers(pimpl->airportMarkers,
                                    airportMarkerCount));

        pimpl->firstFrame = false;
    }

    // Always present text layers (handles GPU buffer uploads).
    for (auto& layer : pimpl->labelLayers) {
        if (layer.text) {
            JST_CHECK(layer.text->present());
        }
    }
    if (pimpl->pathText) {
        JST_CHECK(pimpl->pathText->present());
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
