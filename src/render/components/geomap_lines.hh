#pragma once

#include <cmath>
#include <span>

#include <glm/glm.hpp>

#include "jetstream/render/components/map_context.hh"

namespace Jetstream::Render::Components::GeoMapLines {

// Run exactly the same arc clipping on the CPU (dash layout/tests) and GPU.
using glm::vec3;
using glm::acos;
using glm::atan;
using glm::clamp;
using glm::cos;
using glm::cross;
using glm::dot;
using glm::floor;
using glm::length;
using glm::max;
using glm::min;
using glm::sin;
#include "../../../resources/shaders/map/line_clip.glsl"

// One start/end distance per segment, in logical pixels. Preserve continuity
// across shared endpoints and the antimeridian, resetting only between paths
// or disconnected visible runs. Storage is fixed while GPU buffers refer to it.
inline void UpdateDashRanges(std::span<const F32> vertices,
                             const MapContext& context,
                             std::span<F32> ranges) {
    const auto pixel = context.pixelSize();
    auto project = [&](const glm::vec3& p) {
        const auto clip = context.camera.viewProjection * glm::vec4(p, 1.0f);
        return glm::vec2(clip.x / (clip.w * pixel.x),
                         clip.y / (clip.w * pixel.y));
    };
    F32 distance = 0.0f;
    for (U64 i = 0; i < vertices.size() / 4; ++i) {
        const auto segment = vertices.subspan(i * 4, 4);
        if (i > 0) {
            const F32 longitudeGap = std::abs(segment[0] - vertices[i * 4 - 2]);
            if (segment[1] != vertices[i * 4 - 1] ||
                (longitudeGap != 0.0f && longitudeGap != 360.0f)) {
                distance = 0.0f;
            }
        }
        auto a = MapContext::LonLatToSphere(segment[0], segment[1]);
        auto b = MapContext::LonLatToSphere(segment[2], segment[3]);
        if (!clipLineToHorizon(a, b, glm::vec3(context.camera.targetNormal),
                               context.camera.targetNormal.w)) {
            ranges[i * 2] = ranges[i * 2 + 1] = distance = 0.0f;
            continue;
        }
        ranges[i * 2] = distance;
        distance += glm::length(project(b) - project(a));
        ranges[i * 2 + 1] = distance;
    }
}

}  // namespace Jetstream::Render::Components::GeoMapLines
