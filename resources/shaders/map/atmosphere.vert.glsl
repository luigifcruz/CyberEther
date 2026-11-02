#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "camera.glsl"

layout(location = 0) in vec2 inPosition;

layout(location = 0) out vec3 vWorld;

void main() {
    vWorld = mapSphere(inPosition.xy) * uniforms.lineWidth;
    gl_Position = uniforms.viewProjection * vec4(sphereLocal(vWorld), 1.0);
}
