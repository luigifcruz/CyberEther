#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 transform;
    vec2 thickness;
    float zoom;
    uint numberOfLines;
    vec4 lineInfo;
    vec4 gridColor;
    vec4 majorColor;
} uniforms;

layout(location = 0) in vec4 PosNor; // xy = position, zw = normal

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragNormal;

void main() {
    gl_Position = uniforms.transform * vec4(PosNor.xy, 1.0, 1.0);

    uint lineIndex = gl_VertexIndex / 6u;
    uint numIH = uint(uniforms.lineInfo.x);
    uint numIV = uint(uniforms.lineInfo.y);
    uint numMT = uint(uniforms.lineInfo.z);
    uint numIt = uint(uniforms.lineInfo.w);

    uint interiorEnd = numIH + numIV;
    uint majorTickEnd = interiorEnd + numMT;
    uint minorTickEnd = majorTickEnd + numIt;

    bool isBorder = lineIndex >= minorTickEnd;
    bool isTick = lineIndex >= interiorEnd && lineIndex < minorTickEnd;

    fragColor = (isBorder || isTick) ? uniforms.majorColor.rgb
                                     : uniforms.gridColor.rgb;
    fragNormal = PosNor.zw;
}