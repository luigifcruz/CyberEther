#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    int width;
    int height;
    float index;
    float offset;
    float zoom;
    float panelScaleX;
    float panelScaleY;
    float panelOffsetY;
} uniforms;

layout(set = 0, binding = 1) readonly buffer DataBuffer {
    float data[];
};

layout(set = 0, binding = 2) uniform texture2D lutTex;
layout(set = 0, binding = 3) uniform sampler lutSam;

float sampleWaterfall(float x, float y) {
    int column = clamp(int(floor(x)), 0, uniforms.width - 1);
    int row = int(floor(y)) % uniforms.height;
    if (row < 0) {
        row += uniforms.height;
    }
    return data[row * uniforms.width + column];
}

void main() {
    float magnitude = sampleWaterfall(inTexcoord.x, inTexcoord.y - 4.0) * 0.0162162162;
    magnitude += sampleWaterfall(inTexcoord.x, inTexcoord.y - 3.0) * 0.0540540541;
    magnitude += sampleWaterfall(inTexcoord.x, inTexcoord.y - 2.0) * 0.1216216216;
    magnitude += sampleWaterfall(inTexcoord.x, inTexcoord.y - 1.0) * 0.1945945946;
    magnitude += sampleWaterfall(inTexcoord.x, inTexcoord.y) * 0.2270270270;
    magnitude += sampleWaterfall(inTexcoord.x, inTexcoord.y + 1.0) * 0.1945945946;
    magnitude += sampleWaterfall(inTexcoord.x, inTexcoord.y + 2.0) * 0.1216216216;
    magnitude += sampleWaterfall(inTexcoord.x, inTexcoord.y + 3.0) * 0.0540540541;
    magnitude += sampleWaterfall(inTexcoord.x, inTexcoord.y + 4.0) * 0.0162162162;

    outColor = texture(sampler2D(lutTex, lutSam), vec2(magnitude, 0.0));
}
