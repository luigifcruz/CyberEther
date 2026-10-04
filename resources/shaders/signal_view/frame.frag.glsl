#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inLine;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec4 color;
} uniforms;

void main() {
    float coverage = clamp(inLine.y + 0.5 - abs(inLine.x), 0.0, 1.0);
    outColor = vec4(uniforms.color.rgb, uniforms.color.a * inLine.z * coverage);
}
