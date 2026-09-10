#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"
#include "sun.glsl"

layout(location = 0) in vec2 inQuad;
// Instance: longitude, latitude, urban radius in globe radii, brightness.
layout(location = 1) in vec4 inLight;

layout(location = 0) out vec2 vLocal;
layout(location = 1) out float vIntensity;
layout(location = 2) out float vSeed;

const float HaloScale = 3.5;
const float MinRadiusPixels = 2.5;

void main() {
    vec3 p = lonLatToSphere(inLight.x, inLight.y);
    float facing = dot(p, uniforms.targetNormal.xyz);
    float horizon = uniforms.targetNormal.w;
    float lit = sunLights(p) * sun.lights.x;
    if (facing < horizon || lit < 0.002) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        vLocal = vec2(0.0);
        vIntensity = 0.0;
        vSeed = 0.0;
        return;
    }

    // Screen radius of the built-up core, measured along two surface tangents
    // so the sprite tracks the geography instead of a fixed pixel size.
    vec3 east = cross(vec3(0.0, 1.0, 0.0), p);
    east = length(east) < 1.0e-3 ? vec3(1.0, 0.0, 0.0) : normalize(east);
    vec3 north = cross(p, east);
    vec2 center = projectToNdc(p);
    vec2 pixel = mapPixelSize();
    vec2 eastPx = (projectToNdc(normalize(p + east * inLight.z)) - center) / pixel;
    vec2 northPx = (projectToNdc(normalize(p + north * inLight.z)) - center) / pixel;
    float radiusPx = max(length(eastPx), length(northPx));
    float haloPx = radiusPx * HaloScale;
    float drawnPx = max(haloPx, MinRadiusPixels);

    // Sprites clamped up to the minimum size keep their total light so small
    // towns fade away instead of popping into equal-sized dots.
    float coverage = haloPx / drawnPx;
    coverage *= coverage;
    float limb = smoothstep(horizon, horizon + (1.0 - horizon) * 0.25, facing);

    vLocal = inQuad;
    vIntensity = inLight.w * lit * coverage * limb;
    vSeed = fract(inLight.x * 0.37 + inLight.y * 1.13);
    gl_Position = vec4(center + inQuad * drawnPx * pixel, 0.0, 1.0);
}
