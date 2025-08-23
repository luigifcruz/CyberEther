#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    uint width;
    uint height;
    float offset;
    float zoom;
} uniforms;

layout(set = 0, binding = 1) readonly buffer DataBuffer {
    float data[];
};

layout(set = 0, binding = 2) uniform texture2D lutTex;
layout(set = 0, binding = 3) uniform sampler lutSam;

float samplerXY(float x, float y) {
    if (x < 0.0 || y < 0.0 || x >= uniforms.width || y >= uniforms.height) {
        return 0.0;
    }
    return data[uint(x) + uint(y) * uniforms.width];
}

float cubicHermite(float A, float B, float C, float D, float t) {
    float a = -A/2.0 + (3.0*B)/2.0 - (3.0*C)/2.0 + D/2.0;
    float b = A - (5.0*B)/2.0 + 2.0*C - D / 2.0;
    float c = -A/2.0 + C/2.0;
    float d = B;

    return a*t*t*t + b*t*t + c*t + d;
}

float bicubicInterpolate(float x, float y) {
    float x0 = floor(x) - 1.0;
    float y0 = floor(y) - 1.0;
    float dx = x - (floor(x) - 0.5);
    float dy = y - (floor(y) - 0.5);

    float values[4];
    for (int i = 0; i < 4; i++) {
        values[i] = cubicHermite(
            samplerXY(x0, y0 + i),
            samplerXY(x0 + 1.0, y0 + i),
            samplerXY(x0 + 2.0, y0 + i),
            samplerXY(x0 + 3.0, y0 + i),
            dx
        );
    }

    return cubicHermite(values[0], values[1], values[2], values[3], dy);
}

void main() {
    float x = inTexcoord.x * uniforms.width;
    float y = inTexcoord.y * uniforms.height;

    float color = bicubicInterpolate(x, y);

    outColor = texture(sampler2D(lutTex, lutSam), vec2(color, 0.0f));
}
