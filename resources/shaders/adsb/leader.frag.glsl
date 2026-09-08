#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in float inSide;
layout(location = 1) flat in vec4 inColor;
layout(location = 0) out vec4 outColor;

void main() {
    float alpha = 1.0 - smoothstep(1.0 - max(fwidth(inSide), 0.001), 1.0, abs(inSide));
    outColor = vec4(inColor.rgb, inColor.a * alpha);
}
