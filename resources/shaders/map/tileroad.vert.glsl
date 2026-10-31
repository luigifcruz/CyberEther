#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "line_clip.glsl"
#include "camera.glsl"

layout(location = 0) in vec2 inQuad;
layout(location = 1) in vec4 inSegment;
layout(location = 2) in vec2 inDashRange;

layout(location = 0) out vec2 vLocal;
layout(location = 1) flat out vec2 vShape;
layout(location = 2) flat out vec2 vFade;

void main() {
    vec3 a = lonLatToSphere(inSegment.x, inSegment.y);
    vec3 b = lonLatToSphere(inSegment.z, inSegment.w);
    if (!clipLineToHorizon(a, b, uniforms.targetNormal.xyz,
                          uniforms.targetNormal.w)) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        vLocal = vec2(0.0);
        vShape = vec2(0.0);
        vFade = vec2(0.0);
        return;
    }

    vec2 pixelToNdc = mapPixelSize();
    vec2 start = projectToNdc(a) / pixelToNdc;
    vec2 end = projectToNdc(b) / pixelToNdc;
    vec2 delta = end - start;
    float segmentLength = length(delta);
    vec2 direction = segmentLength > 1e-4 ? delta / segmentLength : vec2(1.0, 0.0);
    vec2 normal = vec2(-direction.y, direction.x);

    float halfWidth = uniforms.lineWidth * 0.5;
    float reach = halfWidth + 1.0;
    float side = inQuad.x * 2.0 - 1.0;
    vec2 current = mix(start, end, inQuad.x) + direction * side * reach +
                   normal * inQuad.y * reach;

    gl_Position = vec4(current * pixelToNdc, 0.0, 1.0);
    vLocal = vec2(inQuad.x * segmentLength + side * reach, inQuad.y * reach);
    vShape = vec2(segmentLength, halfWidth);
    vFade = inDashRange;
}
