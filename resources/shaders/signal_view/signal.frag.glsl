#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec2 thickness;
    float zoom;
    uint numberOfPoints;
    vec4 traceColor;
} uniforms;

layout(location = 0) in vec2 fragNormal;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 color = uniforms.traceColor.rgb;

    // Signed distance to the line's center: 0 at the outer edge, 1 at the center.
    float distance = (1.0 - length(fragNormal));

    // Calculate the gradient of the distance field for anti-aliasing.
    float width = fwidth(distance);
    float edgeSharpness = 0.75;

    // Solid trace with anti-aliased edge (same as the original line).
    float lineAlpha = smoothstep(0.5 - edgeSharpness * width,
                                 0.5 + edgeSharpness * width, distance);

    float haloAlpha = smoothstep(0.0, 0.5, distance) * 0.15;

    outColor = vec4(color, max(lineAlpha, haloAlpha));
}
