#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform FrameUniforms {
    int width;
    int height;
    int channels;
    int useLut;
} uniforms;

layout(set = 0, binding = 1) readonly buffer FrameBuffer {
    float data[];
};

layout(set = 0, binding = 2) uniform texture2D lutTex;
layout(set = 0, binding = 3) uniform sampler lutSam;

void main() {
    int x = clamp(int(inTexcoord.x * float(uniforms.width)), 0, uniforms.width - 1);
    int y = clamp(int((1.0 - inTexcoord.y) * float(uniforms.height)), 0, uniforms.height - 1);
    int base = ((y * uniforms.width) + x) * uniforms.channels;

    float scalar = clamp(data[base], 0.0, 1.0);
    if (uniforms.channels == 1 && uniforms.useLut != 0) {
        outColor = texture(sampler2D(lutTex, lutSam), vec2(scalar, 0.0));
        return;
    }

    if (uniforms.channels == 1) {
        outColor = vec4(vec3(scalar), 1.0);
        return;
    }

    vec3 color = vec3(
        clamp(data[base + 0], 0.0, 1.0),
        clamp(data[base + 1], 0.0, 1.0),
        clamp(data[base + 2], 0.0, 1.0)
    );
    float alpha = (uniforms.channels == 4) ? clamp(data[base + 3], 0.0, 1.0) : 1.0;
    outColor = vec4(color, alpha);
}
