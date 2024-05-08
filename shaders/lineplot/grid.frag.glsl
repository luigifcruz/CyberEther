#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec2 thickness;
    float zoom;
    uint numberOfLines;
} uniforms;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
    // TODO: Implement antialiasing SDF.
}