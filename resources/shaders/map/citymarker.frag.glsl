#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 vLocal;
layout(location = 1) in float vOpacity;

layout(location = 0) out vec4 outColor;

void main() {
    float radius = length(vLocal);
    float aa = max(fwidth(radius), 1.0e-4);
    float background = 1.0 - smoothstep(0.90, 0.90 + aa, radius);
    float outerRing = 1.0 - smoothstep(
        0.075, 0.075 + aa, abs(radius - 0.74));
    float center = 1.0 - smoothstep(0.20, 0.20 + aa, radius);
    float foreground = max(outerRing, center);
    float mask = max(background, foreground);
    if (mask < 0.01) discard;
    vec3 color = mix(vec3(0.01, 0.01, 0.01),
                     vec3(1.0, 0.82, 0.34), foreground);
    outColor = vec4(color, mask * vOpacity);
}
