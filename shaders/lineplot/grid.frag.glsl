#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec2 thickness;
    float zoom;
    uint numberOfLines;
} uniforms;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    // Calculate the distance to the line's edges
    float distance = length(fragNormal);

    // Apply smoothing with a step function
    float alpha = 1.0 - pow(smoothstep(0.0, 1.0, distance), 2);

    // Output the color with the calculated alpha.
    outColor = vec4(fragColor, alpha);
}