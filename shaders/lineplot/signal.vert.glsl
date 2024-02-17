#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
} uniforms;

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec2 texCoord;

void main() {
    gl_Position = uniforms.transform * vec4(inPosition, 1.0);
    texCoord = vec2((inPosition.y + 1.0) / 2.0, 0.0);
}