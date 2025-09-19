#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec2 pixelSize;
    vec2 _padding;
} uniforms;

layout(location = 0) in vec2 inPosition;

// Patch for WebGPU. Implement Naga support for `mat4` scalars.
layout(location = 1) in vec4 inTransform0;
layout(location = 2) in vec4 inTransform1;
layout(location = 3) in vec4 inTransform2;
layout(location = 4) in vec4 inTransform3;

layout(location = 5) in vec4 inFillColor;
layout(location = 6) in vec4 inBorderColor;
layout(location = 7) in vec4 inShapeParams; // x=type, y=borderWidth, z=cornerRadius, w=unused

layout(location = 0) out vec2 outLocalPos;
layout(location = 1) out vec4 outFillColor;
layout(location = 2) out vec4 outBorderColor;
layout(location = 3) out vec4 outShapeParams;
layout(location = 4) out vec2 outPixelSize;

void main() {
    mat4 inTransform = mat4(inTransform0, inTransform1, inTransform2, inTransform3);

    gl_Position = inTransform * vec4(inPosition, 1.0, 1.0);

    // Pass local position for distance calculations in fragment shader
    outLocalPos = inPosition;
    outFillColor = inFillColor;
    outBorderColor = inBorderColor;
    outShapeParams = inShapeParams;
    outPixelSize = uniforms.pixelSize;
}
