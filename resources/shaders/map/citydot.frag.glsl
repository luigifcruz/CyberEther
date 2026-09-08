#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"

layout(location = 0) in vec2 vLocal;
layout(location = 1) in float vOpacity;

layout(location = 0) out vec4 outColor;

void main() {
    float radius = length(vLocal);
    float aa = max(fwidth(radius), 1.0e-4);
    float background = 1.0 - smoothstep(0.90, 0.90 + aa, radius);
    float dot = 1.0 - smoothstep(0.52, 0.52 + aa, radius);
    if (background < 0.01) discard;
    vec3 color = mix(vec3(0.01, 0.01, 0.01),
                     vec3(uniforms.colorR, uniforms.colorG, uniforms.colorB),
                     dot);
    outColor = vec4(color, background * vOpacity);
}
