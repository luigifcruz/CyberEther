#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform texture2D remoteFramebufferTex;
layout(set = 0, binding = 1) uniform sampler remoteFramebufferSam;

void main() {
    outColor = texture(sampler2D(remoteFramebufferTex, remoteFramebufferSam), inTexcoord);
}