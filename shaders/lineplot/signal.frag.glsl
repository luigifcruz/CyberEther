#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
} uniforms;

layout(set = 0, binding = 1) uniform texture2D u_texture;
layout(set = 0, binding = 2) uniform sampler u_sampler;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = texture(sampler2D(u_texture, u_sampler), texCoord).rgb;
    outColor = vec4(color, 1.0);
}
