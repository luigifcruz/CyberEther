#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexcoord;

layout(location = 0) out vec2 outTexcoord;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    uint width;
    uint height;
    float offset;
    float zoom;
} uniforms;

void main() {
    gl_Position = vec4(inPosition, 1.0);
    outTexcoord = inTexcoord;
}
