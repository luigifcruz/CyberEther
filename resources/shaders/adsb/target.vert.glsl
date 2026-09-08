#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ScreenUniforms {
    vec2 pixelSize;
    float lineWidth;
    float padding;
} uniforms;

layout(location = 0) in vec2 inQuad;
layout(location = 1) in vec4 inAnchor; // NDC xy, logical pixel radius, heading
layout(location = 2) in vec4 inColor;
layout(location = 3) in vec4 inStyle; // symbol kind

layout(location = 0) out vec2 outLocal;
layout(location = 1) flat out vec4 outColor;
layout(location = 2) flat out vec2 outStyle;

void main() {
    gl_Position = vec4(inAnchor.xy + inQuad * inAnchor.z * uniforms.pixelSize, 0.0, 1.0);
    outLocal = inQuad;
    outColor = inColor;
    outStyle = vec2(inStyle.x, inAnchor.w);
}
