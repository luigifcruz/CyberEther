#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"

layout(location = 0) in vec2 inPosition;

layout(location = 0) out vec3 vSphere;

void main() {
    vSphere = lonLatToSphere(inPosition.x, inPosition.y);
    gl_Position = uniforms.viewProjection * vec4(vSphere, 1.0);
}
