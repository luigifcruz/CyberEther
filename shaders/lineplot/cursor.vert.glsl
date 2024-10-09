#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
} uniforms;

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec2 fragCoord;

void main() {
    gl_Position = uniforms.transform * vec4(inPosition.xy, 0.0, 1.0);
    fragCoord = inPosition.xy;
}