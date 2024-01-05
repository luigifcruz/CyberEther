#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    uint width;
    uint height;
    float offset;
    float zoom;
} uniforms;

layout(set = 0, binding = 1) uniform texture2D dataTex;
layout(set = 0, binding = 2) uniform sampler dataSam;

layout(set = 0, binding = 3) uniform texture2D lutTex;
layout(set = 0, binding = 4) uniform sampler lutSam;

void main() {
    vec2 color = texture(sampler2D(dataTex, dataSam), inTexcoord).rg;
    outColor = texture(sampler2D(lutTex, lutSam), color);
}