#include "map_layers.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <glm/glm.hpp>

namespace Jetstream::Modules {

namespace {
std::optional<F32> ProjectHeading(const AdsbMapAircraft& aircraft,
                                 const Render::Components::MapContext& context,
                                 const Extent2D<F32>& ndc) {
    if (!aircraft.hasVelocity || !std::isfinite(aircraft.heading)) return std::nullopt;

    const F32 lon = glm::radians(aircraft.longitude);
    const F32 lat = glm::radians(aircraft.latitude);
    const F32 heading = glm::radians(aircraft.heading);
    const glm::vec3 north{-std::sin(lat) * std::sin(lon), std::cos(lat),
                           -std::sin(lat) * std::cos(lon)};
    const glm::vec3 east{std::cos(lon), 0.0f, -std::sin(lon)};
    const glm::vec3 tangent = north * std::cos(heading) + east * std::sin(heading);
    const glm::vec4 direction = context.camera.viewProjection * glm::vec4(tangent, 0.0f);

    // Differentiate clip.xy / clip.w along the local heading. The common
    // positive 1/clip.w factor cancels in atan2; logical pixels account for
    // viewport aspect ratio. This avoids finite-step jitter at high zoom.
    const auto pixel = context.pixelSize();
    const F32 x = (direction.x - ndc.x * direction.w) / pixel.x;
    const F32 y = (direction.y - ndc.y * direction.w) / pixel.y;
    if (!std::isfinite(x) || !std::isfinite(y) || std::hypot(x, y) < 1e-7f) {
        return std::nullopt;
    }
    return std::atan2(x, y);
}
}  // namespace

bool AdsbTargetVisible(const AdsbMapAircraft& aircraft) {
    return !aircraft.positionAgeSeconds ||
           (std::isfinite(*aircraft.positionAgeSeconds) &&
            *aircraft.positionAgeSeconds >= 0 && *aircraft.positionAgeSeconds <= AdsbVisibleSeconds);
}

std::optional<Render::Components::MapContext::Uniforms> AdsbMapAutoFit::update(
    std::span<const AdsbMapAircraft> aircraft,
    const Render::Components::MapContext& context,
    std::span<const MouseEvent> mouseEvents, U64 now) {
    if (!pending) return std::nullopt;
    for (const auto& event : mouseEvents) {
        if (!std::isfinite(event.position.x) || !std::isfinite(event.position.y)) continue;
        if ((event.type == MouseEventType::Click && event.button == MouseButton::Left) ||
            (event.type == MouseEventType::Scroll &&
             std::isfinite(event.scroll.y) && event.scroll.y != 0.0f)) {
            // Yield to map navigation so the view never moves under the pointer.
            pending = false;
            return std::nullopt;
        }
    }

    std::vector<Extent2D<F32>> positions;
    for (const auto& ac : aircraft) {
        if (!AdsbTargetVisible(ac) || !std::isfinite(ac.longitude) ||
            !std::isfinite(ac.latitude) || std::abs(ac.latitude) > 90.0f) continue;
        positions.push_back({ac.longitude, ac.latitude});
    }
    if (positions.empty()) {
        firstPositionTimestamp.reset();
        return std::nullopt;
    }
    if (!firstPositionTimestamp || now < *firstPositionTimestamp) firstPositionTimestamp = now;
    if (now - *firstPositionTimestamp < CollectionMilliseconds) return std::nullopt;

    // Leave regional context around a single plane or a very tight cluster.
    auto view = context.fitView(positions, 0.2f, 5.0f);
    if (view) pending = false;
    return view;
}

std::string FormatAdsbDataBlock(const AdsbMapAircraft& aircraft) {
    std::string identity;
    // The renderer's fonts are ASCII; never let malformed identification text
    // inject extra data-block rows or imply additional decoded fields.
    for (const char c : aircraft.callsign) {
        if (identity.size() == 8) { identity.clear(); break; }
        if (c >= 'a' && c <= 'z') identity += static_cast<char>(c - 'a' + 'A');
        else if ((c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == ' ') identity += c;
        else { identity.clear(); break; }
    }
    while (!identity.empty() && identity.back() == ' ') identity.pop_back();
    if (identity.empty()) identity = jst::fmt::format("ICAO {:06X}", aircraft.icao);

    // Compact terminal-style notation, not a flight level or assigned altitude:
    // reported altitude in hundreds of feet and ground speed in tens of knots.
    const bool altitudeValid = aircraft.hasAltitude && std::isfinite(aircraft.altitude) &&
                               aircraft.altitude >= -9900 && aircraft.altitude <= 99900;
    const bool speedValid = aircraft.hasVelocity && std::isfinite(aircraft.groundSpeed) &&
                            aircraft.groundSpeed >= 0 && aircraft.groundSpeed <= 9990;
    const auto altitude = altitudeValid
        ? jst::fmt::format("{:03d}", std::lround(aircraft.altitude / 100.0f)) : "---";
    const auto speed = speedValid
        ? jst::fmt::format("{:02d}", std::lround(aircraft.groundSpeed / 10.0f)) : "--";
    std::string result = identity + "\n" + altitude + " " + speed;
    if (!aircraft.positionAgeSeconds) result += "\nAGE ?";
    else if (*aircraft.positionAgeSeconds >= AdsbStaleSeconds) {
        result += jst::fmt::format("\nSTALE {:.0f}s", *aircraft.positionAgeSeconds);
    }
    return result;
}

std::vector<AdsbMapTarget> AdsbVisibleTargets(
    std::span<const AdsbMapAircraft> aircraft, const Render::Components::MapContext& context) {
    std::vector<AdsbMapTarget> targets;
    targets.reserve(std::min<U64>(aircraft.size(), AdsbMapState::MaxAircraft));
    for (U64 i = 0; i < aircraft.size() && targets.size() < AdsbMapState::MaxAircraft; ++i) {
        const auto& ac = aircraft[i];
        if (!AdsbTargetVisible(ac)) continue;
        F32 x, y;
        if (!context.projectLonLat(ac.longitude, ac.latitude, x, y) ||
            std::abs(x) > 1.05f || std::abs(y) > 1.05f) continue;
        targets.push_back({i, {x, y}, ProjectHeading(ac, context, {x, y})});
    }
    return targets;
}

std::vector<Extent2D<F32>> AdsbHistoryDots(
    const AdsbMapAircraft& aircraft, const Render::Components::MapContext& context) {
    std::vector<Extent2D<F32>> dots;
    F32 x, y;
    if (!AdsbTargetVisible(aircraft) ||
        !context.projectLonLat(aircraft.longitude, aircraft.latitude, x, y)) return dots;
    Extent2D<F32> previous{x, y};
    const auto pixel = context.pixelSize();
    for (auto it = aircraft.track.rbegin(); it != aircraft.track.rend() && dots.size() < 6; ++it) {
        if (!context.projectLonLat(it->second, it->first, x, y)) continue;
        const F32 distance = std::hypot((x - previous.x) / pixel.x, (y - previous.y) / pixel.y);
        if (distance < 7.0f * AdsbTrackingScale) continue;
        dots.push_back({x, y});
        previous = {x, y};
    }
    return dots;
}

AdsbLabelPlacement PlaceAdsbDataBlock(
    const Extent2D<F32>& anchor, const Extent2D<F32>& size,
    const Extent2D<F32>& viewport, std::span<const AdsbLabelBox> occupied,
    U8 preferredDirection) {
    static constexpr std::array<Extent2D<F32>, 8> directions = {{
        {1, -1}, {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1},
    }};
    AdsbLabelPlacement best{};
    F32 bestScore = std::numeric_limits<F32>::max();
    for (U8 i = 0; i < directions.size(); ++i) {
        const U8 direction = (preferredDirection + i) % directions.size();
        const auto d = directions[direction];
        constexpr F32 gap = 24.0f * AdsbTrackingScale, padding = 6.0f;
        AdsbLabelBox box{
            anchor.x + (d.x > 0 ? gap : d.x < 0 ? -gap - size.x : -size.x * 0.5f),
            anchor.y + (d.y > 0 ? gap : d.y < 0 ? -gap - size.y : -size.y * 0.5f),
            size.x, size.y,
        };
        box.x = std::clamp(box.x, padding, std::max(padding, viewport.x - size.x - padding));
        box.y = std::clamp(box.y, padding, std::max(padding, viewport.y - size.y - padding));
        F32 overlap = 0;
        for (const auto& other : occupied) {
            overlap += std::max(0.0f, std::min(box.x + box.width, other.x + other.width) - std::max(box.x, other.x)) *
                       std::max(0.0f, std::min(box.y + box.height, other.y + other.height) - std::max(box.y, other.y));
        }
        const F32 score = overlap * 1000 + i;
        if (score < bestScore) { best = {box, direction}; bestScore = score; }
        if (overlap == 0) break; // Preserve prior placement whenever it still fits.
    }
    return best;
}

}  // namespace Jetstream::Modules
