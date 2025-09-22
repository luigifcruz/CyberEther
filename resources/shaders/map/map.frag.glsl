#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    float centerLon;
    float centerLat;
    float zoom;
    float aspectRatio;
    float lineWidth;
    float colorR;
    float colorG;
    float colorB;
    float viewportWidth;
    float viewportHeight;
} uniforms;

layout(location = 0) in vec2 vNormal;

layout(location = 0) out vec4 outColor;

void main() {
    float distance = (1.0 - abs(2.0 * vNormal.y - 1.0));

    float width = fwidth(distance);

    float edgeSharpness = 0.75;

    float alpha = smoothstep(0.5 - edgeSharpness * width,
                             0.5 + edgeSharpness * width,
                             distance);

    outColor = vec4(uniforms.colorR, uniforms.colorG, uniforms.colorB,
                    alpha);
}
