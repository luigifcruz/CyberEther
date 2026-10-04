#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in float vertexSlot;

layout(location = 0) out float outElevation;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out float outAge;

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

    bool frequencySide = gl_InstanceIndex == 0;
    float side = frequencySide ? uniforms.skirt.w : uniforms.skirt.z;
    int segments = frequencySide ? uniforms.width - 1 : uniforms.height - 1;
    if (side == 0.0 || segment >= segments) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        outElevation = 0.0;
        outNormal = vec3(0.0, 1.0, 0.0);
        outAge = 0.0;
        return;
    }

    int along = segment + ((corner == 1 || corner == 3 || corner == 4) ? 1 : 0);
    bool top = corner == 2 || corner == 4 || corner == 5;

    int column;
    int row;
    vec3 normal;
    if (frequencySide) {
        column = along;
        row = side > 0.0 ? uniforms.height - 1 : 0;
        normal = vec3(0.0, 0.0, side);
    } else {
        column = side > 0.0 ? uniforms.width - 1 : 0;
        row = along;
        normal = vec3(side, 0.0, 0.0);
    }

    float columnStep = 2.0 / float(max(uniforms.width - 1, 1));
    float rowStep = 2.0 / float(max(uniforms.height - 1, 1));
    float surface = top ? elevation(magnitude(column, row)) : 0.0;

    vec3 position = vec3(float(column) * columnStep - 1.0,
                         surface * uniforms.heightScale,
                         float(row) * rowStep - 1.0);

    gl_Position = uniforms.viewProjection * vec4(position, 1.0);
    outElevation = surface;
    outNormal = normal;
    outAge = 1.0 - float(row) / float(max(uniforms.height - 1, 1));
}
