#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec3 color;
    float sharpness;
} uniforms;

layout(location = 0) in vec2 inTexcoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform texture2D remoteFramebufferTex;
layout(set = 0, binding = 2) uniform sampler remoteFramebufferSam;

void main() {
    // Get the color from the texture.
    float distance = texture(sampler2D(remoteFramebufferTex, remoteFramebufferSam), inTexcoord).r;

    // Calculate the gradient of the distance field.
    float width = fwidth(distance);

    // Convert distance to pixel space.
    float alpha = smoothstep(0.5 - uniforms.sharpness * width, 0.5 + uniforms.sharpness * width, distance);

    // Output the color with the calculated alpha.
    outColor = vec4(uniforms.color, alpha);
}
