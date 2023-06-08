#version 450
#extension GL_ARB_separate_shader_objects : enable

layout (binding = 0) uniform sampler2D lutTextureSampler;

layout(location = 0) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = texture(lutTextureSampler, texCoord).rgb;
    outColor = vec4(color, 1.0);
}
