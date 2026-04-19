#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexcoord;

layout(location = 0) out vec2 outTexcoord;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    uint width;
    uint height;
    float offset;
    float zoom;
    float paddingScaleX;
    float paddingScaleY;
} uniforms;

void main() {
    vec4 position = vec4(inPosition, 1.0);
    position.x *= uniforms.paddingScaleX;
    position.y *= uniforms.paddingScaleY;

    float horizontal = (((inTexcoord.x / uniforms.zoom) + uniforms.offset) * float(uniforms.width));

    gl_Position = position;
    outTexcoord = vec2(horizontal, inTexcoord.y * float(uniforms.height));
}
