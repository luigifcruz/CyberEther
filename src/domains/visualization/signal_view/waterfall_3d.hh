#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_3D_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_3D_HH

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <jetstream/types.hh>
#include <jetstream/logger.hh>
#include <jetstream/surface.hh>
#include <jetstream/render/base.hh>
#include <jetstream/render/colormap.hh>
#include <jetstream/render/components/text.hh>

#include "common.hh"

namespace Jetstream::Modules {

namespace detail {

constexpr F32 kWaterfall3DHeightScale = 0.6f;
constexpr F32 kWaterfall3DFieldOfView = 0.6632f;
constexpr F32 kWaterfall3DNearPlane = 0.05f;
constexpr F32 kWaterfall3DFarPlane = 40.0f;
constexpr F32 kWaterfall3DMinElevation = 0.07f;
constexpr F32 kWaterfall3DMaxElevation = 1.553f;
constexpr F32 kWaterfall3DMinDistance = 1.2f;
constexpr F32 kWaterfall3DMaxDistance = 7.0f;
constexpr F32 kWaterfall3DPanLimit = 1.5f;
constexpr F32 kWaterfall3DOrbitSpeed = 4.7f;
constexpr F32 kWaterfall3DTiltSpeed = 2.4f;
constexpr F32 kWaterfall3DDollySpeed = 0.15f;
constexpr F32 kWaterfall3DSmoothingSeconds = 0.06f;
constexpr F32 kWaterfall3DClickTravel = 0.004f;
constexpr U64 kWaterfall3DMinColumns = 32;
constexpr U64 kWaterfall3DMaxColumns = 768;
constexpr F32 kWaterfall3DPixelsPerColumn = 4.0f;
constexpr F32 kWaterfall3DElevationGain = 4.0f;

struct Waterfall3DCamera {
    F32 azimuth = 0.55f;
    F32 elevation = 0.52f;
    F32 distance = 3.4f;
    glm::vec3 target = {0.0f, 0.15f, 0.0f};

    void clamp() {
        elevation = std::clamp(elevation, kWaterfall3DMinElevation, kWaterfall3DMaxElevation);
        distance = std::clamp(distance, kWaterfall3DMinDistance, kWaterfall3DMaxDistance);
        target.x = std::clamp(target.x, -kWaterfall3DPanLimit, kWaterfall3DPanLimit);
        target.y = std::clamp(target.y, 0.0f, kWaterfall3DHeightScale);
        target.z = std::clamp(target.z, -kWaterfall3DPanLimit, kWaterfall3DPanLimit);
    }

    glm::vec3 eye() const {
        return target + distance * glm::vec3(std::cos(elevation) * std::sin(azimuth),
                                             std::sin(elevation),
                                             std::cos(elevation) * std::cos(azimuth));
    }

    glm::vec3 forward() const {
        return glm::normalize(target - eye());
    }

    glm::vec3 right() const {
        return glm::normalize(glm::cross(forward(), glm::vec3(0.0f, 1.0f, 0.0f)));
    }

    glm::vec3 up() const {
        return glm::cross(right(), forward());
    }

    glm::vec2 span(const F32 aspect) const {
        const F32 vertical = 2.0f * distance * std::tan(kWaterfall3DFieldOfView * 0.5f);
        return {vertical * aspect, vertical};
    }

    void orbit(const F32 deltaAzimuth, const F32 deltaElevation) {
        azimuth += deltaAzimuth;
        elevation += deltaElevation;
        clamp();
    }

    void dolly(const F32 factor) {
        distance *= factor;
        clamp();
    }

    void pan(const glm::vec2& delta, const F32 aspect) {
        const glm::vec2 extent = span(aspect);
        target -= right() * (delta.x * extent.x);
        target += up() * (delta.y * extent.y);
        clamp();
    }

    glm::mat4 view() const {
        return glm::lookAt(eye(), target, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 projection(const F32 aspect) const {
        return glm::perspective(kWaterfall3DFieldOfView,
                                std::max(aspect, 1e-3f),
                                kWaterfall3DNearPlane,
                                kWaterfall3DFarPlane);
    }

    bool approach(const Waterfall3DCamera& goal, const F32 alpha) {
        constexpr F32 epsilon = 1e-4f;
        const F32 weight = std::clamp(alpha, 0.0f, 1.0f);
        bool moved = false;
        const auto step = [&](F32& value, const F32 wanted) {
            const F32 difference = wanted - value;
            if (std::abs(difference) <= epsilon) {
                if (value != wanted) {
                    value = wanted;
                    moved = true;
                }
                return;
            }
            value += difference * weight;
            moved = true;
        };
        step(azimuth, goal.azimuth);
        step(elevation, goal.elevation);
        step(distance, goal.distance);
        step(target.x, goal.target.x);
        step(target.y, goal.target.y);
        step(target.z, goal.target.z);
        return moved;
    }
};

inline U64 Waterfall3DMeshColumns(const U64 bins, const F32 viewWidth) {
    const auto target = static_cast<U64>(std::max(viewWidth / kWaterfall3DPixelsPerColumn, 0.0f));
    return std::min(bins, std::clamp(target, kWaterfall3DMinColumns, kWaterfall3DMaxColumns));
}

inline void Waterfall3DDecimateRow(const F32* bins,
                                   const U64 binCount,
                                   F32* columns,
                                   const U64 columnCount) {
    const F32 step = static_cast<F32>(binCount - 1) / static_cast<F32>(columnCount - 1);
    const I64 last = static_cast<I64>(binCount) - 1;
    for (U64 column = 0; column < columnCount; ++column) {
        const F32 center = static_cast<F32>(column) * step;
        const I64 lo = std::clamp(static_cast<I64>(std::ceil(center - step * 0.5f)), I64{0}, last);
        const I64 hi = std::clamp(static_cast<I64>(std::floor(center + step * 0.5f)), lo, last);
        F32 peak = bins[lo];
        for (I64 bin = lo + 1; bin <= hi; ++bin) {
            peak = std::max(peak, bins[bin]);
        }
        columns[column] = peak;
    }
}

inline F32 Waterfall3DElevation(const F32 magnitude) {
    const F32 gain = kWaterfall3DElevationGain;
    return 0.5f + 0.5f * std::tanh(gain * (magnitude - 0.5f)) / std::tanh(gain * 0.5f);
}

inline F32 Waterfall3DMagnitude(const F32 elevation) {
    const F32 gain = kWaterfall3DElevationGain;
    return 0.5f + std::atanh((2.0f * elevation - 1.0f) * std::tanh(gain * 0.5f)) / gain;
}

inline I32 Waterfall3DSweepOrder(const I32 slot, const I32 count, const F32 cameraIndex) {
    const I32 pivot = std::clamp(static_cast<I32>(std::floor(cameraIndex)), 0, count - 1);
    if (slot < pivot) {
        return slot;
    }
    const I32 remaining = slot - pivot;
    if (remaining < count - 1 - pivot) {
        return count - 1 - remaining;
    }
    return pivot;
}

struct Waterfall3DWalls {
    bool negativeX = false;
    bool positiveX = false;
    bool negativeZ = false;
    bool positiveZ = false;
};

inline Waterfall3DWalls Waterfall3DVisibleWalls(const glm::vec3& eye) {
    return {
        .negativeX = eye.x > -1.0f,
        .positiveX = eye.x < 1.0f,
        .negativeZ = eye.z > -1.0f,
        .positiveZ = eye.z < 1.0f,
    };
}

struct Waterfall3DTick {
    F32 position = 0.0f;
    std::string label;
};

inline std::vector<Waterfall3DTick> Waterfall3DFrequencyTicks(const bool hasFrequency,
                                                              const F32 centerFrequency,
                                                              const F32 sampleRate,
                                                              const U64 count = 5) {
    std::vector<Waterfall3DTick> ticks;
    for (U64 index = 0; index < count; ++index) {
        const F32 position = count > 1 ? -1.0f + 2.0f * index / (count - 1) : 0.0f;
        std::string label;
        if (hasFrequency) {
            label = jst::fmt::format("{:.2f}", (centerFrequency + position * sampleRate * 0.5f) / 1e6f);
        } else {
            label = jst::fmt::format("{:.2f}", (position + 1.0f) * 0.5f);
        }
        ticks.push_back({position, std::move(label)});
    }
    return ticks;
}

inline std::vector<Waterfall3DTick> Waterfall3DTimeTicks(const U64 height, const U64 count = 5) {
    std::vector<Waterfall3DTick> ticks;
    for (U64 index = 0; index < count; ++index) {
        const F32 fraction = count > 1 ? static_cast<F32>(index) / (count - 1) : 0.0f;
        const U64 rows = static_cast<U64>(std::lround(fraction * static_cast<F32>(height)));
        ticks.push_back({1.0f - 2.0f * fraction,
                         rows == 0 ? std::string("0") : jst::fmt::format("-{}", rows)});
    }
    return ticks;
}

inline std::vector<Waterfall3DTick> Waterfall3DAmplitudeTicks(const F32 min,
                                                              const F32 max,
                                                              const U64 count = 5) {
    const F32 lower = std::min(min, max);
    const F32 upper = std::max(min, max);
    const int decimals = upper - lower >= 10.0f ? 0 : 2;
    const F32 epsilon = 0.5f * std::pow(10.0f, static_cast<F32>(-decimals));
    std::vector<Waterfall3DTick> ticks;
    for (U64 index = 0; index < count; ++index) {
        const F32 fraction = count > 1 ? static_cast<F32>(index) / (count - 1) : 0.0f;
        F32 value = lower + Waterfall3DMagnitude(fraction) * (upper - lower);
        if (std::abs(value) < epsilon) {
            value = 0.0f;
        }
        ticks.push_back({fraction, jst::fmt::format("{:.{}f}", value, decimals)});
    }
    return ticks;
}

struct Waterfall3DProjector {
    glm::mat4 viewProjection = glm::mat4(1.0f);
    Extent2D<F32> pixelSize = {0.0f, 0.0f};

    std::optional<glm::vec2> project(const glm::vec3& point) const {
        const glm::vec4 clip = viewProjection * glm::vec4(point, 1.0f);
        if (clip.w <= kWaterfall3DNearPlane) {
            return std::nullopt;
        }
        return glm::vec2(clip.x / clip.w, clip.y / clip.w);
    }

    bool projectSegment(const glm::vec3& a, const glm::vec3& b,
                        glm::vec2& outA, glm::vec2& outB) const {
        glm::vec4 clipA = viewProjection * glm::vec4(a, 1.0f);
        glm::vec4 clipB = viewProjection * glm::vec4(b, 1.0f);
        const F32 limit = kWaterfall3DNearPlane;
        if (clipA.w <= limit && clipB.w <= limit) {
            return false;
        }
        if (clipA.w <= limit) {
            const F32 t = (limit - clipA.w) / (clipB.w - clipA.w);
            clipA = clipA + (clipB - clipA) * t;
        } else if (clipB.w <= limit) {
            const F32 t = (limit - clipB.w) / (clipA.w - clipB.w);
            clipB = clipB + (clipA - clipB) * t;
        }
        outA = glm::vec2(clipA.x / clipA.w, clipA.y / clipA.w);
        outB = glm::vec2(clipB.x / clipB.w, clipB.y / clipB.w);
        return true;
    }
};

constexpr U64 kWaterfall3DGeometryStride = 8;

struct Waterfall3DGeometry {
    std::vector<F32> storage;
    U64 used = 0;

    explicit Waterfall3DGeometry(const U64 capacity = 0)
        : storage(capacity * kWaterfall3DGeometryStride, 0.0f) {}

    U64 capacity() const {
        return storage.size() / kWaterfall3DGeometryStride;
    }

    void clear() {
        used = 0;
    }

    void vertex(const glm::vec2& position, const F32 distancePx,
                const F32 halfWidthPx, const F32 alpha) {
        if (used >= capacity()) {
            return;
        }
        F32* slot = storage.data() + used * kWaterfall3DGeometryStride;
        slot[0] = position.x;
        slot[1] = position.y;
        slot[2] = distancePx;
        slot[3] = halfWidthPx;
        slot[4] = alpha;
        slot[5] = 0.0f;
        slot[6] = 0.0f;
        slot[7] = 0.0f;
        ++used;
    }

    void quad(const glm::vec2& a, const glm::vec2& b,
              const glm::vec2& c, const glm::vec2& d, const F32 alpha) {
        if (used + 6 > capacity()) {
            return;
        }
        constexpr F32 solid = 1e4f;
        vertex(a, 0.0f, solid, alpha);
        vertex(b, 0.0f, solid, alpha);
        vertex(c, 0.0f, solid, alpha);
        vertex(a, 0.0f, solid, alpha);
        vertex(c, 0.0f, solid, alpha);
        vertex(d, 0.0f, solid, alpha);
    }

    void line(const glm::vec2& a, const glm::vec2& b,
              const Extent2D<F32>& pixelSize, const F32 widthPx, const F32 alpha) {
        line(a, b, pixelSize, widthPx, alpha, alpha);
    }

    void line(const glm::vec2& a, const glm::vec2& b,
              const Extent2D<F32>& pixelSize, const F32 widthPx,
              const F32 alphaA, const F32 alphaB) {
        if (used + 6 > capacity()) {
            return;
        }
        const glm::vec2 scale(pixelSize.x, pixelSize.y);
        const glm::vec2 direction = (b - a) / scale;
        const F32 length = glm::length(direction);
        if (!(length > 1e-4f)) {
            return;
        }
        const F32 halfWidth = 0.5f * widthPx;
        const F32 reach = halfWidth + 1.0f;
        const glm::vec2 normal = glm::vec2(-direction.y, direction.x) / length;
        const glm::vec2 offset = normal * reach * scale;
        vertex(a + offset, reach, halfWidth, alphaA);
        vertex(a - offset, -reach, halfWidth, alphaA);
        vertex(b + offset, reach, halfWidth, alphaB);
        vertex(a - offset, -reach, halfWidth, alphaA);
        vertex(b - offset, -reach, halfWidth, alphaB);
        vertex(b + offset, reach, halfWidth, alphaB);
    }
};

struct Waterfall3DAxisFrame {
    bool valid = false;
    glm::vec2 a = {0.0f, 0.0f};
    glm::vec2 b = {0.0f, 0.0f};
    glm::vec2 direction = {1.0f, 0.0f};
    glm::vec2 normal = {0.0f, -1.0f};
};

inline Waterfall3DAxisFrame Waterfall3DAxisFrameFor(const Waterfall3DProjector& projector,
                                                    const glm::vec3& start,
                                                    const glm::vec3& finish,
                                                    const glm::vec3& outward,
                                                    const glm::vec2& centerNdc) {
    Waterfall3DAxisFrame frame;
    const auto a = projector.project(start);
    const auto b = projector.project(finish);
    if (!a || !b) {
        return frame;
    }
    const glm::vec2 scale(projector.pixelSize.x, projector.pixelSize.y);
    const glm::vec2 direction = (*b - *a) / scale;
    const F32 length = glm::length(direction);
    if (!(length > 1e-3f)) {
        return frame;
    }
    frame.valid = true;
    frame.a = *a;
    frame.b = *b;
    frame.direction = direction / length;

    const glm::vec3 middle = 0.5f * (start + finish);
    const glm::vec2 middleNdc = 0.5f * (*a + *b);
    glm::vec2 normal(0.0f);
    if (const auto pushed = projector.project(middle + outward * 0.05f)) {
        normal = (*pushed - middleNdc) / scale;
    }
    const F32 normalLength = glm::length(normal);
    if (normalLength > 1e-3f) {
        normal /= normalLength;
    } else {
        normal = glm::vec2(-frame.direction.y, frame.direction.x);
        if (glm::dot(normal, (middleNdc - centerNdc) / scale) < 0.0f) {
            normal = -normal;
        }
    }
    frame.normal = normal;
    return frame;
}

inline F32 Waterfall3DLabelExtentAlong(const glm::vec2& normal,
                                       const Extent2D<I32>& alignment,
                                       const F32 widthPx,
                                       const F32 heightPx) {
    const F32 x0 = alignment.x == 0 ? 0.0f : (alignment.x == 1 ? -0.5f * widthPx : -widthPx);
    const F32 y0 = alignment.y == 0 ? -heightPx : (alignment.y == 1 ? -0.5f * heightPx : 0.0f);
    F32 extent = 0.0f;
    for (const F32 x : {x0, x0 + widthPx}) {
        for (const F32 y : {y0, y0 + heightPx}) {
            extent = std::max(extent, glm::dot(glm::vec2(x, y), normal));
        }
    }
    return extent;
}

inline U64 Waterfall3DTickStride(const Waterfall3DAxisFrame& frame,
                                 const Extent2D<F32>& pixelSize,
                                 const U64 tickCount,
                                 const F32 labelWidthPx,
                                 const F32 lineHeightPx) {
    if (!frame.valid || tickCount < 2) {
        return 1;
    }
    const glm::vec2 scale(pixelSize.x, pixelSize.y);
    const F32 spacing = glm::length((frame.b - frame.a) / scale) / static_cast<F32>(tickCount - 1);
    const F32 needed = std::abs(frame.direction.x) * labelWidthPx +
                       std::abs(frame.direction.y) * lineHeightPx + 6.0f;
    if (spacing >= needed) {
        return 1;
    }
    if (spacing * 2.0f >= needed) {
        return 2;
    }
    return tickCount - 1;
}

}  // namespace detail

struct SignalViewWaterfall3DLabels {
    std::string frequency;
    std::string time;
    std::string amplitude;
    bool hasFrequency = false;
    F32 centerFrequency = 0.0f;
    F32 sampleRate = 0.0f;
    F32 rangeMin = 0.0f;
    F32 rangeMax = 1.0f;

    bool operator==(const SignalViewWaterfall3DLabels&) const = default;
};

class SignalViewWaterfall3D {
 public:
    Result create(const std::shared_ptr<Render::Window>& window,
                  U64 width,
                  U64 height,
                  const std::string& colormap);
    Result destroy(const std::shared_ptr<Render::Window>& window);

    Result present(std::vector<SurfaceEvent>&& surfaceEvents,
                   std::vector<InputEvent>&& inputEvents,
                   const WaterfallFrame& frame,
                   const SignalViewWaterfall3DLabels& labels,
                   const std::string& colormap,
                   bool& viewChanged);

    const std::shared_ptr<Render::Texture>& framebuffer() const {
        return framebufferTexture;
    }

    const Extent2D<U64>& viewSize() const {
        return interaction.viewSize;
    }

    bool created() const {
        return static_cast<bool>(renderSurface);
    }

 private:
    struct MeshUniforms {
        glm::mat4 viewProjection;
        glm::vec4 cameraCell;
        glm::vec4 lightDirection;
        glm::vec4 fade;
        glm::vec4 background;
        glm::vec4 viewport;
        I32 width;
        I32 height;
        I32 writeIndex;
        F32 heightScale;
        glm::vec4 skirt;
    };

    struct FrameUniforms {
        glm::vec4 color;
    };

    struct FrameLayer {
        detail::Waterfall3DGeometry geometry;
        std::shared_ptr<Render::Buffer> verticesBuffer;
        std::shared_ptr<Render::Vertex> vertex;
        std::shared_ptr<Render::Draw> draw;
        std::shared_ptr<Render::Program> program;

        explicit FrameLayer(const U64 capacity) : geometry(capacity) {}
    };

    struct TraceLayer {
        std::shared_ptr<Render::Vertex> vertex;
        std::shared_ptr<Render::Draw> draw;
        std::shared_ptr<Render::Program> program;
    };

    struct DragState {
        bool orbiting = false;
        bool panning = false;
        F32 travel = 0.0f;
        glm::vec2 last = {0.0f, 0.0f};
    };

    struct AmplitudeAxis {
        glm::vec2 corner = {-1.0f, 1.0f};
        F32 side = -1.0f;
    };

    U64 width = 0;
    U64 height = 0;
    U64 columns = 0;
    U64 columnCapacity = 0;
    std::vector<F32> heights;
    U64 writeIndex = 0;
    SignalViewWaterfall3DLabels labels;

    SurfaceInteractionState interaction;
    detail::Waterfall3DCamera camera;
    detail::Waterfall3DCamera cameraGoal;
    DragState drag;
    std::chrono::steady_clock::time_point lastFrameTime;
    bool clockStarted = false;
    bool sceneDirty = true;
    bool ticksDirty = true;

    std::vector<detail::Waterfall3DTick> frequencyTicks;
    std::vector<detail::Waterfall3DTick> timeTicks;
    std::vector<detail::Waterfall3DTick> amplitudeTicks;

    Extent2D<F32> pixelSize;
    MeshUniforms meshUniforms{};
    FrameUniforms frameUniforms{};
    std::vector<F32> meshSlots;

    std::shared_ptr<Render::Texture> framebufferTexture;
    Render::Colormap lut;
    std::shared_ptr<Render::Surface> renderSurface;
    std::shared_ptr<Render::Components::Text> text;

    std::shared_ptr<Render::Buffer> meshSlotsBuffer;
    std::shared_ptr<Render::Buffer> meshUniformBuffer;
    std::shared_ptr<Render::Buffer> heightsBuffer;
    std::shared_ptr<Render::Vertex> meshVertex;
    std::shared_ptr<Render::Draw> drawMesh;
    std::shared_ptr<Render::Program> meshProgram;

    std::shared_ptr<Render::Vertex> skirtVertex;
    std::shared_ptr<Render::Draw> drawSkirt;
    std::shared_ptr<Render::Program> skirtProgram;

    TraceLayer traceBehind;
    TraceLayer traceInFront;

    std::shared_ptr<Render::Buffer> frameUniformBuffer;
    FrameLayer backdrop{1536};
    FrameLayer foreground{96};

    Result resizeMesh(const F32* bins, U64 nextColumns);
    Result uploadRows(const F32* bins, U64 startRow, U64 rowCount);
    void processInputEvents(std::vector<InputEvent>&& events);
    bool advanceCamera();
    void updateTicks();
    Result updateScene();
    Result updateFrameLayer(FrameLayer& layer);
    AmplitudeAxis amplitudeAxis(const detail::Waterfall3DProjector& projector,
                                const glm::vec3& eye) const;
    void updateLabels(const detail::Waterfall3DProjector& projector,
                      const glm::vec3& eye,
                      const AmplitudeAxis& amplitude);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_WATERFALL_3D_HH
