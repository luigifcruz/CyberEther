#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"

layout(location = 0) in vec2 vNormal;
layout(location = 1) in float vDashCoord;

layout(location = 0) out vec4 outColor;

void main() {
    float distance = (1.0 - abs(2.0 * vNormal.y - 1.0));
    float width = fwidth(distance);
    float edgeSharpness = 0.75;
    float alpha = smoothstep(0.0, edgeSharpness * width, distance);

    if (uniforms.lineStyle > 0.5) {
        float coord = vDashCoord / max(uniforms.dashScale, 1.0) +
                      uniforms.dashPhase;
        float pattern = fract(coord);
        float aa = max(fwidth(coord), 1.0e-4);
        float pulseDistance = min(pattern, 0.5 - pattern);
        alpha *= smoothstep(-aa, aa, pulseDistance);
    }

    outColor = vec4(uniforms.colorR, uniforms.colorG, uniforms.colorB,
                    alpha * uniforms.lineOpacity);
}
