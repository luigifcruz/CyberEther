#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
} uniforms;

layout(location = 0) in vec2 fragCoord;

layout(location = 0) out vec4 outColor;

void main() {
    float radius = 0.5;
    float thickness = 0.05;
    float distance = length(fragCoord);

    float edge0 = radius - thickness;
    float edge1 = radius + thickness;
    float alpha = smoothstep(edge0, edge1, distance);

    outColor = vec4(1.0, 1.0, 1.0, (1.0 - alpha) - 0.10);
}