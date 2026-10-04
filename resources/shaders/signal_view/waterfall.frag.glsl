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
    int filtered;
} uniforms;

layout(set = 0, binding = 1) readonly buffer DataBuffer {
    float data[];
};

layout(set = 0, binding = 2) readonly buffer FilteredBuffer {
    float filtered[];
};

layout(set = 0, binding = 3) uniform texture2D lutTex;
layout(set = 0, binding = 4) uniform sampler lutSam;

float sampleWaterfall(float x, float y, int offset) {
    int column = clamp(int(floor(x)), 0, uniforms.width - 1);
    int writeIndex = int(round(uniforms.index * float(uniforms.height)));
    int age = int(floor(y)) - (writeIndex - uniforms.height);
    age = clamp(age + offset, 0, uniforms.height - 1);
    int row = (writeIndex + age) % uniforms.height;
    return data[row * (uniforms.width + 16) + column];
}

void main() {
    float y = inTexcoord.y;
    float row = floor(y);
    if (uniforms.filtered != 0 &&
        floor(y - 4.0) == row - 4.0 && floor(y + 4.0) == row + 4.0) {
        int column = clamp(int(floor(inTexcoord.x)), 0, uniforms.width - 1);
        int wrapped = int(row) % uniforms.height;
        if (wrapped < 0) {
            wrapped += uniforms.height;
        }
        float magnitude = filtered[wrapped * uniforms.width + column];
        float mapped = 0.5 + 0.5 * tanh(4.0 * (magnitude - 0.5));
        outColor = texture(sampler2D(lutTex, lutSam), vec2(mapped, 0.0));
        return;
    }
    float magnitude = sampleWaterfall(inTexcoord.x, y, -4) * 0.0162162162;
    magnitude += sampleWaterfall(inTexcoord.x, y, -3) * 0.0540540541;
    magnitude += sampleWaterfall(inTexcoord.x, y, -2) * 0.1216216216;
    magnitude += sampleWaterfall(inTexcoord.x, y, -1) * 0.1945945946;
    magnitude += sampleWaterfall(inTexcoord.x, y, 0) * 0.2270270270;
    magnitude += sampleWaterfall(inTexcoord.x, y, 1) * 0.1945945946;
    magnitude += sampleWaterfall(inTexcoord.x, y, 2) * 0.1216216216;
    magnitude += sampleWaterfall(inTexcoord.x, y, 3) * 0.0540540541;
    magnitude += sampleWaterfall(inTexcoord.x, y, 4) * 0.0162162162;

    float mapped = 0.5 + 0.5 * tanh(4.0 * (magnitude - 0.5));
    outColor = texture(sampler2D(lutTex, lutSam), vec2(mapped, 0.0));
}
