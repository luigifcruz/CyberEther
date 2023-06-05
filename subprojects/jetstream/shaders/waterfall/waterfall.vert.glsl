#version 450

layout(location = 0) in vec3 vertexArray;
layout(location = 1) in vec2 texcoord;
layout(location = 0) out vec2 outTexCoord;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    int width;
    int height;
    int maxSize;
    float index;
    float offset;
    float zoom;
    bool interpolate;
} uniforms;

void main() {
    gl_Position = vec4(vertexArray, 1.0);
    float vertical = ((uniforms.index - (1.0 - texcoord.y)) * float(uniforms.height));
    float horizontal = (((texcoord.x / uniforms.zoom) + uniforms.offset) * float(uniforms.width));
    outTexCoord = vec2(horizontal, vertical);
}
