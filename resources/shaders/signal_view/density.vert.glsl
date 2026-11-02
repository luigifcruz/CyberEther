#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vertexArray;
layout(location = 1) in vec2 texcoord;

layout(location = 0) out vec2 outTexcoord;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    uint columns;
    uint levels;
    float binsPerColumn;
    float binSpan;
    float offset;
    float zoom;
    float panelScaleX;
    float panelScaleY;
    float plotWidth;
    float intensityScale;
    float panelOffsetY;
} uniforms;

void main() {
    vec4 position = vec4(vertexArray, 1.0);
    position.x *= uniforms.panelScaleX;
    position.y = position.y * uniforms.panelScaleY + uniforms.panelOffsetY;

    float bin = ((texcoord.x / uniforms.zoom) + uniforms.offset) * uniforms.binSpan;
    outTexcoord = vec2((bin + 0.5) / uniforms.binsPerColumn,
                       texcoord.y * float(uniforms.levels));
    gl_Position = position;
}
