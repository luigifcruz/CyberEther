#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform StarsUniforms {
    mat4 skyViewProjection;  // view-projection with camera translation stripped
    float viewportWidth;
    float viewportHeight;
    float surfaceScale;
    float _pad;
} uniforms;

layout(location = 0) in vec2 inQuad;   // -1..1 quad corner
layout(location = 1) in vec4 inStar;   // xyz = unit direction, w = brightness 0..1

layout(location = 0) out vec2 vLocal;
layout(location = 1) out float vBrightness;
layout(location = 2) out float vSeed;

void main() {
    vec4 clip = uniforms.skyViewProjection * vec4(inStar.xyz, 1.0);

    // Cull stars behind the camera (on the far half of the celestial sphere).
    if (clip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        vLocal = vec2(0.0);
        vBrightness = 0.0;
        vSeed = 0.0;
        return;
    }

    vec2 center = clip.xy / clip.w;

    // Apparent size in pixels grows with brightness; convert to NDC.
    float sizePx = mix(1.5, 7.0, inStar.w);
    vec2 pixelToNdc = vec2(
        (2.0 * uniforms.surfaceScale) / max(uniforms.viewportWidth, 1.0),
        (2.0 * uniforms.surfaceScale) / max(uniforms.viewportHeight, 1.0)
    );

    vLocal = inQuad;
    vBrightness = inStar.w;
    // Stable per-star seed from direction for color variation.
    vSeed = fract(dot(inStar.xyz, vec3(12.9898, 78.233, 37.719)) * 43758.5453);

    gl_Position = vec4(center + inQuad * sizePx * pixelToNdc, 0.0, 1.0);
}
