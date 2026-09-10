#include "jetstream/render/components/map_context.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "jetstream/logger.hh"

namespace Jetstream::Render::Components {

namespace {
constexpr F32 HalfFovSin = 0.382683432365f;
constexpr F32 HalfFovTan = 0.414213562373f;
constexpr F32 BaseHeight = 1.0f / HalfFovSin - 1.0f;
constexpr F32 MinScreenFraction = 0.70f;
constexpr F32 MinZoom = -8.0f;
constexpr F32 MaxLatitude = 85.05112878f;

F32 FitZoom(const MapContext::Uniforms& view) {
    return std::min(std::log2(std::max(view.aspectRatio, 1.0f)), MapContext::MaxZoom);
}

F32 MinDetailZoom(const MapContext::Uniforms& view) {
    const F32 tanLimit = std::min(view.aspectRatio, 1.0f) * HalfFovTan;
    const F32 distance = 1.0f / std::sin(std::atan(MinScreenFraction * tanLimit));
    return std::log2(BaseHeight / (distance - 1.0f)) - FitZoom(view);
}

void SphereToLonLat(const glm::vec3& p, F32& lon, F32& lat) {
    lat = glm::degrees(std::asin(std::clamp(p.y, -1.0f, 1.0f)));
    lon = glm::degrees(std::atan2(p.x, p.z));
}

Result UpdateNavigation(MapContext::Uniforms view, MapContext& context) {
    const F32 fitZoom = FitZoom(view);
    view.detailZoom = std::clamp(view.detailZoom, MinDetailZoom(view),
                                 MapContext::MaxZoom - fitZoom);
    view.zoom = view.detailZoom + fitZoom;
    return context.update(view);
}
}  // namespace

static_assert(sizeof(MapContext::GpuUniforms) == 144,
              "Map uniforms must match the std140 ShaderUniforms block.");

MapContext::MapContext() {
    update(view);
}

glm::vec3 MapContext::LonLatToSphere(F32 lon, F32 lat) {
    const F32 r = glm::radians(lat);
    const F32 lr = glm::radians(lon);
    const F32 cl = std::cos(r);
    return {cl * std::sin(lr), std::sin(r), cl * std::cos(lr)};
}

glm::vec3 MapContext::CameraUp(const glm::vec3& n) {
    const glm::vec3 proj = glm::vec3(0.0f, 1.0f, 0.0f) - n * n.y;
    const F32 len = glm::length(proj);
    if (len < 1e-4f) {
        return n.y > 0.0f ? glm::vec3(0.0f, 0.0f, -1.0f)
                         : glm::vec3(0.0f, 0.0f, 1.0f);
    }
    return proj / len;
}

F32 MapContext::WrapLongitude(F32 longitude) {
    F32 wrapped = std::fmod(longitude + 180.0f, 360.0f);
    if (wrapped < 0.0f) wrapped += 360.0f;
    return wrapped - 180.0f;
}

Extent2D<F64> MapContext::SubsolarPoint(F64 unixSeconds) {
    const auto wrapDegrees = [](F64 degrees) {
        degrees = std::fmod(degrees, 360.0);
        return degrees < 0.0 ? degrees + 360.0 : degrees;
    };
    const F64 days = unixSeconds / 86400.0 - 10957.5;
    const F64 meanLongitude = glm::radians(wrapDegrees(280.460 + 0.9856474 * days));
    const F64 meanAnomaly = glm::radians(wrapDegrees(357.528 + 0.9856003 * days));
    const F64 eclipticLongitude = meanLongitude +
        glm::radians(1.915) * std::sin(meanAnomaly) +
        glm::radians(0.020) * std::sin(2.0 * meanAnomaly);
    const F64 obliquity = glm::radians(23.439 - 0.0000004 * days);
    const F64 declination = std::asin(std::sin(obliquity) * std::sin(eclipticLongitude));
    const F64 rightAscension = std::atan2(std::cos(obliquity) * std::sin(eclipticLongitude),
                                          std::cos(eclipticLongitude));
    const F64 siderealTime = glm::radians(wrapDegrees(280.46061837 + 360.98564736629 * days));
    const F64 longitude = wrapDegrees(glm::degrees(rightAscension - siderealTime) + 180.0) - 180.0;
    return {longitude, glm::degrees(declination)};
}

glm::vec3 MapContext::SunDirection(F64 unixSeconds) {
    const auto point = SubsolarPoint(unixSeconds);
    return LonLatToSphere(static_cast<F32>(point.x), static_cast<F32>(point.y));
}

Result MapContext::update(const Uniforms& uniforms) {
    if (!std::isfinite(uniforms.centerLon) ||
        !std::isfinite(uniforms.centerLat) ||
        !std::isfinite(uniforms.zoom) ||
        uniforms.zoom < MinZoom || uniforms.zoom > MaxZoom ||
        !std::isfinite(uniforms.detailZoom) ||
        uniforms.detailZoom < MinZoom || uniforms.detailZoom > MaxZoom ||
        !std::isfinite(uniforms.surfaceScale) ||
        uniforms.surfaceScale <= 0.0f || uniforms.surfaceScale > 64.0f) {
        JST_ERROR("[GEOMAP] Map uniforms are outside valid ranges.");
        return Result::ERROR;
    }

    view = uniforms;
    view.centerLon = WrapLongitude(view.centerLon);
    view.centerLat = std::clamp(view.centerLat, -MaxLatitude, MaxLatitude);
    view.viewportWidth = std::isfinite(view.viewportWidth)
        ? std::max(view.viewportWidth, 1.0f) : 1.0f;
    view.viewportHeight = std::isfinite(view.viewportHeight)
        ? std::max(view.viewportHeight, 1.0f) : 1.0f;
    if (!std::isfinite(view.aspectRatio) ||
        view.aspectRatio < std::numeric_limits<F32>::epsilon()) {
        view.aspectRatio = view.viewportWidth / view.viewportHeight;
    }

    const F32 height = BaseHeight / std::pow(2.0f, view.zoom);
    const F32 distance = 1.0f + height;
    const glm::vec3 normal = LonLatToSphere(view.centerLon, view.centerLat);
    const glm::vec3 position = distance * normal;
    const glm::vec3 up = CameraUp(normal);
    const glm::mat4 projection = glm::perspective(
        static_cast<F32>(45.0 * JST_PI / 180.0), view.aspectRatio,
        std::max(1e-7f, height * 1e-3f), distance + 2.0f);
    camera = {};
    camera.viewProjection = projection *
        glm::lookAt(position, glm::vec3(0.0f), up);
    skyViewProjection = projection *
        glm::lookAt(glm::vec3(0.0f), -normal, up);
    camera.cameraPos = glm::vec4(position, 1.0f);
    camera.targetNormal = glm::vec4(normal, 1.0f / distance);
    camera.surfaceScale = view.surfaceScale;
    camera.viewportWidth = view.viewportWidth;
    camera.viewportHeight = view.viewportHeight;
    return Result::SUCCESS;
}

std::optional<MapContext::Uniforms> MapContext::fitView(
    std::span<const Extent2D<F32>> points, F32 padding, F32 maxZoom) const {
    if (!std::isfinite(padding) || padding < 0.0f || padding >= 1.0f ||
        !std::isfinite(maxZoom) || maxZoom < MinZoom || maxZoom > MaxZoom) {
        return std::nullopt;
    }

    std::vector<Extent2D<F32>> valid;
    F32 minLat = 90.0f;
    F32 maxLat = -90.0f;
    for (const auto& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y) ||
            std::abs(point.y) > 90.0f) continue;
        valid.push_back({WrapLongitude(point.x), point.y});
        minLat = std::min(minLat, point.y);
        maxLat = std::max(maxLat, point.y);
    }
    if (valid.empty()) return std::nullopt;

    // The complement of the largest circular gap is the shortest longitude
    // interval containing every point, including clusters straddling +/-180.
    std::sort(valid.begin(), valid.end(), [](const auto& a, const auto& b) {
        return a.x < b.x;
    });
    F32 largestGap = -1.0f;
    F32 startLon = valid.front().x;
    for (U64 i = 0; i < valid.size(); ++i) {
        const F32 next = i + 1 < valid.size() ? valid[i + 1].x : valid.front().x + 360.0f;
        const F32 gap = next - valid[i].x;
        if (gap > largestGap) {
            largestGap = gap;
            startLon = next;
        }
    }

    auto fitted = view;
    fitted.centerLon = WrapLongitude(startLon + (360.0f - largestGap) * 0.5f);
    fitted.centerLat = (minLat + maxLat) * 0.5f;
    const F32 fitZoom = FitZoom(fitted);
    F32 low = std::max({MinZoom, MinZoom + fitZoom, MinDetailZoom(fitted) + fitZoom});
    F32 high = maxZoom;
    if (low > high) return std::nullopt;

    MapContext candidate;
    const auto fits = [&](F32 zoom) {
        fitted.zoom = zoom;
        fitted.detailZoom = zoom - fitZoom;
        if (candidate.update(fitted) != Result::SUCCESS) return false;
        for (const auto& point : valid) {
            F32 x, y;
            if (!candidate.projectLonLat(point.x, point.y, x, y) ||
                std::abs(x) > 1.0f - padding || std::abs(y) > 1.0f - padding) {
                return false;
            }
        }
        return true;
    };

    // Test the actual perspective projection and horizon, rather than treating
    // longitude/latitude bounds as a flat map. A globe cannot show every span.
    if (!fits(low)) return std::nullopt;
    if (fits(high)) return candidate.view;
    for (U32 i = 0; i < 24; ++i) {
        const F32 mid = (low + high) * 0.5f;
        if (fits(mid)) low = mid;
        else high = mid;
    }
    if (!fits(low)) return std::nullopt;
    return candidate.view;
}

bool MapContext::projectLonLat(F32 lon, F32 lat, F32& ndcX, F32& ndcY) const {
    if (!std::isfinite(lon) || !std::isfinite(lat) || std::abs(lat) > 90.0f) {
        return false;
    }
    const glm::vec3 p = LonLatToSphere(lon, lat);
    if (glm::dot(p, glm::vec3(camera.targetNormal)) < camera.targetNormal.w) {
        return false;
    }
    const glm::vec4 clip = camera.viewProjection * glm::vec4(p, 1.0f);
    if (clip.w <= 0.0f) return false;
    ndcX = clip.x / clip.w;
    ndcY = clip.y / clip.w;
    return true;
}

Extent2D<F32> MapContext::pixelSize() const {
    return {(2.0f * view.surfaceScale) / view.viewportWidth,
            (2.0f * view.surfaceScale) / view.viewportHeight};
}

Result MapNavigation::resize(const SurfaceEvent& event, MapContext& context) {
    if (event.type != SurfaceEventType::Resize) return Result::SUCCESS;
    auto view = context.view;
    view.viewportWidth = std::max(static_cast<F32>(event.size.x), 1.0f);
    view.viewportHeight = std::max(static_cast<F32>(event.size.y), 1.0f);
    view.aspectRatio = view.viewportWidth / view.viewportHeight;
    view.surfaceScale = event.scale;
    return UpdateNavigation(view, context);
}

Result MapNavigation::mouse(const MouseEvent& event, MapContext& context) {
    auto view = context.view;
    switch (event.type) {
        case MouseEventType::Scroll: {
            if (!std::isfinite(event.scroll.y)) return Result::SUCCESS;
            const F32 fitZoom = FitZoom(view);
            const F32 newZoom = std::clamp(
                view.detailZoom + event.scroll.y * 0.15f,
                MinDetailZoom(view), MapContext::MaxZoom - fitZoom);
            if (newZoom == view.detailZoom) break;
            const F32 oldHeight = BaseHeight / std::pow(2.0f, view.detailZoom + fitZoom);
            const F32 newHeight = BaseHeight / std::pow(2.0f, newZoom + fitZoom);
            const glm::vec3 normal = MapContext::LonLatToSphere(view.centerLon, view.centerLat);
            const glm::vec3 north = MapContext::CameraUp(normal);
            const glm::vec3 east = glm::normalize(glm::cross(north, normal));
            const F32 x = (event.position.x - 0.5f) * 2.0f;
            const F32 y = (0.5f - event.position.y) * 2.0f;
            const glm::vec3 offset = (x * view.aspectRatio * HalfFovTan) * east +
                                     (y * HalfFovTan) * north;
            SphereToLonLat(glm::normalize(normal + (oldHeight - newHeight) * offset),
                           view.centerLon, view.centerLat);
            view.detailZoom = newZoom;
            break;
        }
        case MouseEventType::Click:
            if (event.button == MouseButton::Left) {
                dragging = true;
                dragAnchor = event.position;
                dragStartLon = view.centerLon;
                dragStartLat = view.centerLat;
            }
            break;
        case MouseEventType::Move:
            if (dragging) {
                const F32 height = BaseHeight / std::pow(2.0f, view.detailZoom + FitZoom(view));
                const F32 x = (event.position.x - dragAnchor.x) * 2.0f;
                const F32 y = (dragAnchor.y - event.position.y) * 2.0f;
                const glm::vec3 normal = MapContext::LonLatToSphere(dragStartLon, dragStartLat);
                const glm::vec3 north = MapContext::CameraUp(normal);
                const glm::vec3 east = glm::normalize(glm::cross(north, normal));
                SphereToLonLat(glm::normalize(normal -
                    (x * height * view.aspectRatio * HalfFovTan) * east -
                    (y * height * HalfFovTan) * north), view.centerLon, view.centerLat);
            }
            break;
        case MouseEventType::Release:
        case MouseEventType::Leave:
            cancel();
            break;
        default:
            break;
    }
    return UpdateNavigation(view, context);
}

void MapNavigation::cancel() {
    dragging = false;
}

}  // namespace Jetstream::Render::Components
