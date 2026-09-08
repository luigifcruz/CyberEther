#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ScreenUniforms {
    vec2 pixelSize;
    float lineWidth;
    float padding;
} uniforms;

layout(location = 0) in vec2 inQuad;
layout(location = 1) in vec4 inEndpoints;
layout(location = 2) in vec4 inColor;
layout(location = 0) out float outSide;
layout(location = 1) flat out vec4 outColor;

void main() {
    vec2 direction = (inEndpoints.zw - inEndpoints.xy) / uniforms.pixelSize;
    float distance = length(direction);
    vec2 perpendicular = distance > 1e-7 ? vec2(-direction.y, direction.x) / distance : vec2(0.0);
    vec2 offset = perpendicular * uniforms.pixelSize * uniforms.lineWidth * 0.5 * inQuad.y;
    gl_Position = vec4(mix(inEndpoints.xy, inEndpoints.zw, (inQuad.x + 1.0) * 0.5) + offset, 0.0, 1.0);
    outSide = inQuad.y;
    outColor = inColor;
}
