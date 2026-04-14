#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    float centerLon;
    float centerLat;
    float zoom;
    float aspectRatio;
    float surfaceScale;
    float lineWidth;
    float colorR;
    float colorG;
    float colorB;
    float viewportWidth;
    float viewportHeight;
} uniforms;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(uniforms.colorR, uniforms.colorG, uniforms.colorB,
                    1.0);
}
