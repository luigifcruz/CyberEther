#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "line_clip.glsl"
#include "camera.glsl"

// Quad vertex: x = endpoint selector (0 or 1), y = side (-1 or +1).
layout(location = 0) in vec2 inQuad;

// Instance data: (lon1, lat1, lon2, lat2) in degrees.
layout(location = 1) in vec4 inSegment;
// Continuous projected distance along a reference polyline (logical pixels).
layout(location = 2) in vec2 inDashRange;

layout(location = 0) out vec2 vNormal;
layout(location = 1) out float vDashCoord;

void main() {
    vec3 a = lonLatToSphere(inSegment.x, inSegment.y);
    vec3 b = lonLatToSphere(inSegment.z, inSegment.w);
    if (!clipLineToHorizon(a, b, uniforms.targetNormal.xyz,
                          uniforms.targetNormal.w)) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        vNormal = vec2(0.0);
        vDashCoord = 0.0;
        return;
    }

    vec2 start = projectToNdc(a);
    vec2 end = projectToNdc(b);
    vec2 current = mix(start, end, inQuad.x);

    vec2 direction = end - start;
    float segmentLength = length(direction);
    if (segmentLength < 1e-7) {
        gl_Position = vec4(current, 0.0, 1.0);
        vNormal = vec2(0.0, (inQuad.y + 1.0) * 0.5);
        vDashCoord = 0.0;
        return;
    }

    direction /= segmentLength;
    vec2 perpendicular = vec2(-direction.y, direction.x);
    vec2 pixelToNdc = mapPixelSize();
    vec2 thickness = vec2(uniforms.lineWidth * 0.5) * pixelToNdc;
    vec2 offset = perpendicular * thickness * inQuad.y;

    gl_Position = vec4(current + offset, 0.0, 1.0);
    vNormal = vec2(0.0, (inQuad.y + 1.0) * 0.5);

    vDashCoord = mix(inDashRange.x, inDashRange.y, inQuad.x);
}
