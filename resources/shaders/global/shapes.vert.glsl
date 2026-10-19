#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec2 pixelSize;
    vec2 translation;
} uniforms;

layout(location = 0) in vec2 inPosition;

// Patch for WebGPU. Implement Naga support for `mat4` scalars.
layout(location = 1) in vec4 inTransform0;
layout(location = 2) in vec4 inTransform1;
layout(location = 3) in vec4 inTransform2;
layout(location = 4) in vec4 inTransform3;

layout(location = 5) in vec4 inFillColor;
layout(location = 6) in vec4 inBorderColor;
layout(location = 7) in vec4 inShapeParams; // x=type, y=borderWidth, z=cornerRadius, w=corner mask (0 = all)
layout(location = 8) in vec4 inHole; // xy=local origin, zw=local size (empty = no hole)

layout(location = 0) out vec2 outLocalPos;
layout(location = 1) out vec4 outFillColor;
layout(location = 2) out vec4 outBorderColor;
layout(location = 3) out vec4 outShapeParams;
layout(location = 4) out vec2 outPixelSize;

const int TYPE_TRIANGLE = 0;

void main() {
    mat4 inTransform = mat4(inTransform0, inTransform1, inTransform2, inTransform3);

    vec2 corner = sign(inPosition);
    vec2 localPos;
    if (abs(inPosition.x) > 1.5) {
        bool hollow = inHole.z > 0.0 && inHole.w > 0.0;
        localPos = hollow ? inHole.xy + inHole.zw * (corner * 0.5 + 0.5) : vec2(0.0);
    } else if (int(inShapeParams.x) == TYPE_TRIANGLE) {
        localPos = corner;
    } else {
        vec2 pixel = max(abs(uniforms.pixelSize), vec2(1.0e-12));
        vec2 sizePixels = vec2(length(inTransform0.xy / pixel), length(inTransform1.xy / pixel));
        localPos = corner * (0.5 + 1.0 / max(sizePixels, vec2(1.0e-6)));
    }

    gl_Position = inTransform * vec4(localPos, 1.0, 1.0);
    gl_Position.xy += uniforms.translation * gl_Position.w;

    // Pass local position for distance calculations in fragment shader
    outLocalPos = localPos;
    outFillColor = inFillColor;
    outBorderColor = inBorderColor;
    outShapeParams = inShapeParams;
    outPixelSize = uniforms.pixelSize;
}
