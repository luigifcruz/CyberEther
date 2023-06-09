#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;
layout(location = 1) flat in uint inMaxSize;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    uint width;
    uint height;
    float offset;
    float zoom;
} uniforms;

layout(set = 0, binding = 1) buffer DataBuffer {
    float data[];
};

layout(set = 0, binding = 2) uniform sampler2D lut;

void main() {
    uvec2 texcoord = uvec2(inTexcoord);
    uint index = texcoord.y * uniforms.width + texcoord.x;

    if (index > inMaxSize || index < 0) {
        outColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        outColor = texture(lut, vec2(data[index], 0.0));
    }
}
