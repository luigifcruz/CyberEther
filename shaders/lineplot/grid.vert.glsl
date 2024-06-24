#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec2 thickness;
    float zoom;
    uint numberOfLines;
} uniforms;

layout(location = 0) in vec4 PosNor; // xy = position, zw = normal

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragNormal;

void main() {
    gl_Position = uniforms.transform * vec4(PosNor.xy, 1.0, 1.0);
    fragColor = vec3(0.2, 0.2, 0.2);
    fragNormal = PosNor.zw;
}