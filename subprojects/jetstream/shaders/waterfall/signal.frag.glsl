#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    int width;
    int height;
    int maxSize;
    float index;
    float offset;
    float zoom;
    bool interpolate;
} uniforms;

layout(set = 0, binding = 1) readonly buffer DataBuffer {
    float data[];
};

layout(set = 0, binding = 2) uniform sampler2D lut;

float samplerXY(float x, float y) {
    int idx = int(y) * uniforms.width + int(x);
    if (idx < uniforms.maxSize && idx > 0) {
        return data[idx];
    } else {
        idx += uniforms.maxSize;
        if (idx < uniforms.maxSize && idx > 0) {
            return data[idx];
        } else {
            return 1.0;
        }
    }
}

void main() {
    float mag = 0.0;

    if (uniforms.interpolate) {
        mag += samplerXY(inTexcoord.x, inTexcoord.y - 4.0) * 0.0162162162;
        mag += samplerXY(inTexcoord.x, inTexcoord.y - 3.0) * 0.0540540541;
        mag += samplerXY(inTexcoord.x, inTexcoord.y - 2.0) * 0.1216216216;
        mag += samplerXY(inTexcoord.x, inTexcoord.y - 1.0) * 0.1945945946;
        mag += samplerXY(inTexcoord.x, inTexcoord.y) * 0.2270270270;
        mag += samplerXY(inTexcoord.x, inTexcoord.y + 1.0) * 0.1945945946;
        mag += samplerXY(inTexcoord.x, inTexcoord.y + 2.0) * 0.1216216216;
        mag += samplerXY(inTexcoord.x, inTexcoord.y + 3.0) * 0.0540540541;
        mag += samplerXY(inTexcoord.x, inTexcoord.y + 4.0) * 0.0162162162;
    } else {
        mag = samplerXY(inTexcoord.x, inTexcoord.y);
    }

    outColor = texture(lut, vec2(mag, 0.0));
}