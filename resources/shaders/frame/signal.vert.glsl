#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexcoord;

layout(location = 0) out vec2 outTexcoord;

layout(set = 0, binding = 0) uniform FrameUniforms {
    int width;
    int height;
    int channels;
    int useLut;
    int interpolate;
    float rangeMin;
    float rangeScale;
    float zoom;
    float centerX;
    float centerY;
    float fitScaleX;
    float fitScaleY;
    float paddingScaleX;
    float paddingScaleY;
} uniforms;

void main() {
    vec4 position = vec4(inPosition, 1.0);
    position.x *= uniforms.paddingScaleX;
    position.y *= uniforms.paddingScaleY;

    gl_Position = position;
    outTexcoord = inTexcoord;
}
