#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec3 color;
} uniforms;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexcoord;

layout(location = 0) out vec2 outTexcoord;

void main() {
    gl_Position = uniforms.transform * vec4(inPosition, 1.0, 1.0);
    outTexcoord = inTexcoord;
}
