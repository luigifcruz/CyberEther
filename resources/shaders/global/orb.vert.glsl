#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec4 rect;
    vec4 orb;
    vec4 params;
    vec4 colorA;
    vec4 colorB;
    vec4 colorC;
    vec4 colorD;
} uniforms;

layout(location = 0) in vec2 inPosition;

layout(location = 0) out vec2 outUv;

void main() {
    vec2 ndc = uniforms.rect.xy + inPosition * uniforms.rect.zw;
    gl_Position = vec4(ndc, 1.0, 1.0);
    outUv = inPosition;
}
