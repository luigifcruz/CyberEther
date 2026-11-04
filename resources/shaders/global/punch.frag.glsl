#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec4 frame;
    vec4 rect;
    vec4 params;
} uniforms;

layout(location = 0) in vec2 inTexcoord;
layout(location = 1) in vec2 inLocal;

layout(location = 0) out vec4 outColor;

void main() {
    vec2 halfSize = uniforms.rect.zw * 0.5;
    float radius = clamp(uniforms.params.x, 0.0, min(halfSize.x, halfSize.y));
    vec2 q = abs(inLocal - halfSize) - (halfSize - vec2(radius));
    float distance = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
    float coverage = clamp(0.5 - distance, 0.0, 1.0);

    outColor = vec4(0.0, 0.0, 0.0, coverage * uniforms.params.y);
}
