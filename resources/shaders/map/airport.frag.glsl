#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 vLocal;
layout(location = 1) in float vOpacity;

layout(location = 0) out vec4 outColor;

void main() {
    float radius = length(vLocal);
    float aa = max(fwidth(radius), 1.0e-4);
    float ring = 1.0 - smoothstep(0.08, 0.08 + aa, abs(radius - 0.72));
    float cross = 1.0 - smoothstep(0.08, 0.08 + aa,
                                  min(abs(vLocal.x), abs(vLocal.y)));
    float mask = max(ring, cross * (1.0 - smoothstep(0.55, 0.65, radius)));
    if (mask < 0.01) discard;
    outColor = vec4(0.95, 0.78, 0.40, mask * vOpacity);
}
