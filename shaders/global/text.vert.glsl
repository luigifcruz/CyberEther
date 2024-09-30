#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec3 color;
    float sharpness;
} uniforms;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexcoord;

// TODO: Patch for WebGPU. Implement Naga support for `mat4` scalars.
layout(location = 2) in vec4 inTransform0;
layout(location = 3) in vec4 inTransform1;
layout(location = 4) in vec4 inTransform2;
layout(location = 5) in vec4 inTransform3;

layout(location = 0) out vec2 outTexcoord;

void main() {
    mat4 inTransform = mat4(inTransform0, inTransform1, inTransform2, inTransform3);
    gl_Position = inTransform * vec4(inPosition, 1.0, 1.0);
    outTexcoord = inTexcoord;
}
