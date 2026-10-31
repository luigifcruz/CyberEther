#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inFade;

layout(location = 0) out vec3 vColor;
layout(location = 1) out vec3 vSphere;
layout(location = 2) flat out vec2 vFade;

void main() {
    vec3 p = lonLatToSphere(inPosition.x, inPosition.y);
    gl_Position = uniforms.viewProjection * vec4(p, 1.0);
    vColor = inColor;
    vSphere = p;
    vFade = inFade;
}
