#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec4 frame;
    vec4 rect;
    vec4 params;
} uniforms;

layout(location = 0) in vec2 inPosition;

layout(location = 0) out vec2 outTexcoord;
layout(location = 1) out vec2 outLocal;

void main() {
    vec2 size = max(uniforms.frame.xy, vec2(1.0));
    vec2 position = uniforms.frame.zw + uniforms.rect.xy + inPosition * uniforms.rect.zw;
    gl_Position = vec4(position.x / size.x * 2.0 - 1.0,
                       1.0 - position.y / size.y * 2.0,
                       1.0,
                       1.0);
    outTexcoord = inPosition;
    outLocal = inPosition * uniforms.rect.zw;
}
