#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"

layout(location = 0) in vec2 vLocal;
layout(location = 1) flat in vec2 vShape;
layout(location = 2) flat in vec2 vFade;

layout(location = 0) out vec4 outColor;

void main() {
    float along = clamp(vLocal.x, 0.0, vShape.x);
    float distance = uniforms.lineStyle > 0.5
        ? max(abs(vLocal.x - along), abs(vLocal.y))
        : length(vec2(vLocal.x - along, vLocal.y));
    float alpha = clamp(vShape.y - distance + 0.5, 0.0, 1.0);
    alpha *= tileFade(vFade);
    if (alpha <= 0.0) {
        discard;
    }
    outColor = vec4(uniforms.colorR, uniforms.colorG, uniforms.colorB,
                    alpha * uniforms.lineOpacity);
}
