#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec2 thickness;
    float zoom;
    uint numberOfPoints;
    vec4 traceColor;
} uniforms;

layout(location = 0) in vec2 inPosition;

layout(location = 0) out float vY;

void main() {
    gl_Position = uniforms.transform * vec4(inPosition, 1.0, 1.0);
    vY = inPosition.y;
}
