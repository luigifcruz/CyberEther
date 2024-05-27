#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec2 thickness;
    float zoom;
    uint numberOfPoints;
} uniforms;

layout(set = 0, binding = 1) uniform texture2D u_texture;
layout(set = 0, binding = 2) uniform sampler u_sampler;

layout(location = 0) in vec2 texCoord;
layout(location = 1) in vec2 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    // Get the color from the texture.
    vec3 color = texture(sampler2D(u_texture, u_sampler), texCoord).rgb;

    // Calculate the distance to the line's edges
    float distance = length(fragNormal);

    // Apply smoothing with a step function
    float alpha = smoothstep(0.0, 1.0, distance);

    outColor = vec4(color, 1.0 - alpha);
}
