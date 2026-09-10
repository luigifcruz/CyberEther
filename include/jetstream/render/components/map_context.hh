#ifndef JETSTREAM_RENDER_COMPONENTS_MAP_CONTEXT_HH
#define JETSTREAM_RENDER_COMPONENTS_MAP_CONTEXT_HH

#include <optional>
#include <span>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "jetstream/types.hh"
#include "jetstream/surface.hh"

namespace Jetstream::Render::Components {

struct JETSTREAM_API MapContext {
    static constexpr F32 MaxZoom = 12.0f;

    struct Uniforms {
        float centerLon = 0.0f;
        float centerLat = 0.0f;
        float zoom = 0.0f;
        float detailZoom = 0.0f;
        float aspectRatio = 1.0f;
        float surfaceScale = 1.0f;
        float viewportWidth = 800.0f;
        float viewportHeight = 600.0f;

        bool operator==(const Uniforms&) const = default;
    } view;

    struct GpuUniforms {
        glm::mat4 viewProjection;
        glm::vec4 cameraPos;
        glm::vec4 targetNormal;
        float surfaceScale;
        float viewportWidth;
        float viewportHeight;
        float lineWidth;
        float colorR;
        float colorG;
        float colorB;
        float lineStyle;
        float dashScale;
        float dashPhase;
        float outlineStrength;
        float lineOpacity;
    } camera{};

    glm::mat4 skyViewProjection{1.0f};
    std::optional<Extent2D<F32>> cursor;

    MapContext();
    Result update(const Uniforms& uniforms);

    std::optional<Uniforms> fitView(std::span<const Extent2D<F32>> points,
                                    F32 padding = 0.2f, F32 maxZoom = MaxZoom) const;
    bool projectLonLat(F32 lon, F32 lat, F32& ndcX, F32& ndcY) const;
    Extent2D<F32> pixelSize() const;

    static glm::vec3 LonLatToSphere(F32 lon, F32 lat);
    static glm::vec3 CameraUp(const glm::vec3& normal);
    static F32 WrapLongitude(F32 longitude);
    static Extent2D<F64> SubsolarPoint(F64 unixSeconds);
    static glm::vec3 SunDirection(F64 unixSeconds);
};

class JETSTREAM_API MapNavigation {
 public:
    Result resize(const SurfaceEvent& event, MapContext& context);
    Result mouse(const MouseEvent& event, MapContext& context);
    void cancel();

 private:
    bool dragging = false;
    Extent2D<F32> dragAnchor{};
    F32 dragStartLon = 0.0f;
    F32 dragStartLat = 0.0f;
};

}  // namespace Jetstream::Render::Components

#endif
