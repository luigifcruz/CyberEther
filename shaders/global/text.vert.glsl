#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec3 color;
} uniforms;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexcoord;
layout(location = 2) in mat4 inTransform;

layout(location = 0) out vec2 outTexcoord;

void main() {
    gl_Position = inTransform * vec4(inPosition, 1.0, 1.0);
    outTexcoord = inTexcoord;
}
