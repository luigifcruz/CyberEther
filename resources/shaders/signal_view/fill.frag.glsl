#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec2 thickness;
    float zoom;
    uint numberOfPoints;
    vec4 traceColor;
} uniforms;

layout(location = 0) in float vY;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(uniforms.traceColor.rgb, uniforms.traceColor.a);
}
