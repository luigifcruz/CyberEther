#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec2 thickness;
    float zoom;
    uint numberOfPoints;
} uniforms;

layout(location = 0) in vec4 PosNor; // xy = position, zw = normal  

layout(location = 0) out vec2 texCoord;
layout(location = 1) out vec2 fragNormal;

void main() {
    gl_Position = uniforms.transform * vec4(PosNor.xy, 1.0, 1.0);
    texCoord = vec2((PosNor.y + 1.0) / 2.0, 0.0);
    fragNormal = PosNor.zw;
}