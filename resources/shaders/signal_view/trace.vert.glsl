#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in float vertexSlot;

layout(location = 0) out vec2 outLine;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 viewProjection;
    vec4 cameraCell;
    vec4 lightDirection;
    vec4 fade;
    vec4 background;
    vec4 viewport;
    int width;
    int height;
    int writeIndex;
    float heightScale;
    vec4 skirt;
} uniforms;

layout(set = 0, binding = 1) readonly buffer DataBuffer {
    float data[];
};

float fetch(int column, int row) {
    column = clamp(column, 0, uniforms.width - 1);
    row = clamp(row, 0, uniforms.height - 1);
    int bufferRow = (uniforms.writeIndex + row) % uniforms.height;
    return data[bufferRow * uniforms.width + column];
}

float magnitude(int column, int row) {
    return clamp(fetch(column, row), 0.0, 1.0);
}

float elevation(float value) {
    return 0.5 + 0.5 * tanh(4.0 * (value - 0.5)) / tanh(2.0);
}

void main() {
    int slot = int(vertexSlot + 0.5);
    int segment = slot / 6;
    int corner = slot - segment * 6;
    int row = uniforms.height - 1;
    float columnStep = 2.0 / float(max(uniforms.width - 1, 1));

    vec3 p0 = vec3(float(segment) * columnStep - 1.0,
                   elevation(magnitude(segment, row)) * uniforms.heightScale, 1.0);
    vec3 p1 = vec3(float(segment + 1) * columnStep - 1.0,
                   elevation(magnitude(segment + 1, row)) * uniforms.heightScale, 1.0);
    vec4 clip0 = uniforms.viewProjection * vec4(p0, 1.0);
    vec4 clip1 = uniforms.viewProjection * vec4(p1, 1.0);

    float halfWidth = uniforms.viewport.z * 0.5;
    float reach = halfWidth + 3.0;
    vec2 pixels = uniforms.viewport.xy * 0.5;
    vec2 n0 = clip0.xy / clip0.w;
    vec2 n1 = clip1.xy / clip1.w;
    vec2 direction = (n1 - n0) * pixels;
    float length2 = dot(direction, direction);
    if (clip0.w <= 0.05 || clip1.w <= 0.05 || length2 < 1e-8) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        outLine = vec2(0.0, halfWidth);
        return;
    }
    vec2 normal = vec2(-direction.y, direction.x) * inversesqrt(length2);

    bool atEnd = corner == 2 || corner == 4 || corner == 5;
    float side = (corner == 0 || corner == 2 || corner == 5) ? 1.0 : -1.0;
    vec2 base = atEnd ? n1 : n0;
    gl_Position = vec4(base + normal * (side * reach) / pixels, 0.0, 1.0);
    outLine = vec2(side * reach, halfWidth);
}
