#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 inShape;
layout(location = 1) in vec4 inStyle;

layout(location = 0) out vec4 outLine;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec4 color;
    vec4 lineColor;
} uniforms;

void main() {
    gl_Position = vec4(inShape.xy, 0.0, 1.0);
    outLine = vec4(inShape.z, inShape.w, inStyle.x, inStyle.y);
}
