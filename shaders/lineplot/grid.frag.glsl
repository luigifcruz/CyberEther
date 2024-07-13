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
    // Calculate the distance to the line's center.
    float distance = 0.5 - (1.0 - abs(2.0 * fragNormal.y - 1.0));

    // Calculate the gradient of the distance field.
    vec2 ddist = vec2(dFdx(distance), dFdy(distance));

    // Convert distance to pixel space.
    float pixelDist = distance / length(ddist);

    // Apply anti-aliasing.
    float alpha = clamp(0.5 - pixelDist, 0.0, 1.0);

    // Output the color with the calculated alpha.
    outColor = vec4(fragColor, alpha);
}