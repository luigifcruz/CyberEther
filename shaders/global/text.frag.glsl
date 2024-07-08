#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec3 color;
} uniforms;

layout(location = 0) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform texture2D remoteFramebufferTex;
layout(set = 0, binding = 2) uniform sampler remoteFramebufferSam;

void main() {
    float distance = texture(sampler2D(remoteFramebufferTex, remoteFramebufferSam), inTexcoord).r;
    float alpha = smoothstep(0.5 - 0.1, 0.5 + 0.1, distance);
    outColor = vec4(uniforms.color, alpha);
}