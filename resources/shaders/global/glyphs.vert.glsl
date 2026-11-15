#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    vec2 pixelSize;
    float atlasPixelRange;
    float em;
    vec4 reserved;
} uniforms;

layout(location = 0) in vec4 inQuad;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec4 inLine;
layout(location = 3) in vec4 inPlace;
layout(location = 4) in vec4 inStyle;

layout(location = 0) out vec2 outTexcoord;
layout(location = 1) out vec4 outColor;
layout(location = 2) out float outLineY;
layout(location = 3) out vec4 outStyle;

void main() {
    vec2 pixel = inQuad.xy + inPlace.xy;
    gl_Position = vec4(pixel.x * uniforms.pixelSize.x - 1.0,
                       1.0 - pixel.y * uniforms.pixelSize.y,
                       1.0,
                       1.0);
    outTexcoord = inQuad.zw;
    outColor = inColor;
    outLineY = inLine.x;
    outStyle = vec4(inPlace.z, inPlace.w, inStyle.x, inStyle.y);
}
