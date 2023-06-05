#version 450

layout(location = 0) in vec3 vertexArray;
layout(location = 0) out vec2 texcoord;

void main() {
    float pos = (vertexArray.y + 1.0) / 2.0;
    texcoord = vec2(pos, 0.0);
    gl_Position = vec4(vertexArray.x, vertexArray.y, vertexArray.z, 1.0);
}