#version 450

layout (std140, binding = 0) uniform ShaderUniforms {
    int width;
    int height;
    int maxSize;
    float index;
    float offset;
    float zoom;
    bool interpolate;
} uniforms;

layout (binding = 1) uniform sampler2D lut;

layout (location = 0) in vec2 texcoord;

layout (location = 0) out vec4 outColor;

void main() {
    float mag = 0.0;
    int _idx;

    if (uniforms.interpolate) {

    } else {
    }

    outColor = texture(lut, vec2(mag, 0.0));
}