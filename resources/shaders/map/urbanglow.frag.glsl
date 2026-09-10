#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"
#include "sun.glsl"

layout(location = 0) in vec3 vSphere;

layout(location = 0) out vec4 outColor;

float hash3(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1030, 0.0973));
    p += dot(p, p.yxz + 33.33);
    return fract((p.x + p.y) * p.z);
}

float valueNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float x0 = mix(hash3(i + vec3(0.0, 0.0, 0.0)), hash3(i + vec3(1.0, 0.0, 0.0)), u.x);
    float x1 = mix(hash3(i + vec3(0.0, 1.0, 0.0)), hash3(i + vec3(1.0, 1.0, 0.0)), u.x);
    float x2 = mix(hash3(i + vec3(0.0, 0.0, 1.0)), hash3(i + vec3(1.0, 0.0, 1.0)), u.x);
    float x3 = mix(hash3(i + vec3(0.0, 1.0, 1.0)), hash3(i + vec3(1.0, 1.0, 1.0)), u.x);
    return mix(mix(x0, x1, u.y), mix(x2, x3, u.y), u.z);
}

void main() {
    vec3 normal = normalize(vSphere);
    if (dot(normal, uniforms.targetNormal.xyz) < uniforms.targetNormal.w) {
        discard;
    }

    float lit = sunLights(normal) * sun.lights.y;
    if (lit < 0.003) {
        discard;
    }

    // Procedural lighting texture in a stable domain on the unit globe, about
    // one unit per 1.5 km. Octaves finer than the pixel footprint collapse to
    // their mean so the texture never shimmers while panning zoomed out.
    vec3 q = normal * 4200.0;
    float footprint = max(fwidth(q.x), max(fwidth(q.y), fwidth(q.z)));
    float pattern = 0.0;
    float total = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    for (int octave = 0; octave < 4; ++octave) {
        float keep = 1.0 - smoothstep(0.25, 0.75, footprint * frequency);
        pattern += amplitude * mix(0.5, valueNoise(q * frequency), keep);
        total += amplitude;
        amplitude *= 0.5;
        frequency *= 2.3;
    }
    pattern /= total;
    float density = smoothstep(0.30, 0.85, pattern);

    // Individual bright points at street scale, only once they are resolved.
    vec3 fine = q * 6.0;
    float fineFootprint = footprint * 6.0;
    float sparkle = 0.0;
    if (fineFootprint < 0.5) {
        vec3 cell = floor(fine);
        vec3 jitter = vec3(hash3(cell + 0.37), hash3(cell + 1.91),
                           hash3(cell + 4.73)) * 0.7 + 0.15;
        float spacing = length(fract(fine) - jitter);
        sparkle = step(0.72, hash3(cell)) * exp(-spacing * spacing * 40.0) *
                  (1.0 - smoothstep(0.25, 0.5, fineFootprint));
    }

    // Sodium orange with a pale core, drifting cooler in some regions.
    vec3 sodium = vec3(1.00, 0.62, 0.26);
    vec3 pale = vec3(1.00, 0.93, 0.74);
    vec3 cool = vec3(0.86, 0.92, 1.00);
    float regional = smoothstep(0.55, 0.85, valueNoise(normal * 9.0));
    vec3 base = mix(sodium, cool, regional * 0.45);
    vec3 color = mix(base, pale, clamp(density * 0.6 + sparkle, 0.0, 1.0));
    float alpha = lit * clamp(0.20 + 0.40 * density + 0.35 * sparkle, 0.0, 1.0);
    outColor = vec4(color, alpha);
}
