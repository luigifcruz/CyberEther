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

layout(set = 0, binding = 1) uniform sampler2D data;
layout(set = 0, binding = 2) uniform sampler2D lut;

void main() {
    vec2 color = texture(data, inTexcoord).rb;
    outColor = texture(lut, color);
}