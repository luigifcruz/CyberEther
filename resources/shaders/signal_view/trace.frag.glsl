#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inLine;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 viewProjection;
    vec4 cameraCell;
    vec4 lightDirection;
    vec4 fade;
    vec4 background;
    vec4 viewport;
    int width;
    int height;
    int writeIndex;
    float heightScale;
    vec4 skirt;
} uniforms;

void main() {
    float distance = abs(inLine.x);
    float core = clamp(inLine.y + 0.5 - distance, 0.0, 1.0);
    float halo = clamp(1.0 - (distance - inLine.y) / 3.0, 0.0, 1.0) * 0.25;
    outColor = vec4(1.0, 1.0, 1.0, max(core, halo));
}
