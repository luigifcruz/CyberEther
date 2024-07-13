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
    // Get the color from the texture.
    float distance = 0.5 - texture(sampler2D(remoteFramebufferTex, remoteFramebufferSam), inTexcoord).r;
    
    // Calculate the gradient of the distance field.
    vec2 ddist = vec2(dFdx(distance), dFdy(distance));
    
    // Convert distance to pixel space.
    float pixelDist = distance / length(ddist);
    
    // Apply anti-aliasing.
    float alpha = clamp(0.5 - pixelDist, 0.0, 1.0);

    // Output the color with the calculated alpha.
    outColor = vec4(uniforms.color, alpha);
}
