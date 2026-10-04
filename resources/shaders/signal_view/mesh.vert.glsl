#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in float vertexSlot;

layout(location = 0) out float outMagnitude;
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

int sweepOrder(int slot, int count, float cameraIndex) {
    int pivot = clamp(int(floor(cameraIndex)), 0, count - 1);
    if (slot < pivot) {
        return slot;
    }
    int remaining = slot - pivot;
    if (remaining < count - 1 - pivot) {
        return count - 1 - remaining;
    }
    return pivot;
}

float fetch(int column, int row) {
    column = clamp(column, 0, uniforms.width - 1);
    row = clamp(row, 0, uniforms.height - 1);
    int bufferRow = (uniforms.writeIndex + row) % uniforms.height;
    return data[bufferRow * uniforms.width + column];
}

float magnitude(int column, int row) {
    float value = fetch(column, row - 2) * 0.0625;
    value += fetch(column, row - 1) * 0.25;
    value += fetch(column, row) * 0.375;
    value += fetch(column, row + 1) * 0.25;
    value += fetch(column, row + 2) * 0.0625;
    return clamp(value, 0.0, 1.0);
}

float elevation(float value) {
    return 0.5 + 0.5 * tanh(4.0 * (value - 0.5)) / tanh(2.0);
}

float surfaceHeight(int column, int row) {
    return elevation(magnitude(column, row)) * uniforms.heightScale;
}

void main() {
    int cellsPerRow = uniforms.width - 1;
    int slot = int(vertexSlot + 0.5);
    int cellSlot = slot / 6;
    int corner = slot - cellSlot * 6;

    int column = sweepOrder(cellSlot, cellsPerRow, uniforms.cameraCell.x);
    int row = sweepOrder(gl_InstanceIndex, uniforms.height - 1, uniforms.cameraCell.y);

    float towardCamera = (uniforms.cameraCell.x - float(column) - 0.5) +
                         (uniforms.cameraCell.y - float(row) - 0.5);
    if (towardCamera < 0.0) {
        corner = (corner + 3) % 6;
    }

    int dc = (corner == 1 || corner == 3 || corner == 4) ? 1 : 0;
    int dr = (corner == 2 || corner == 4 || corner == 5) ? 1 : 0;
    column += dc;
    row += dr;

    float value = magnitude(column, row);
    float columnStep = 2.0 / float(max(uniforms.width - 1, 1));
    float rowStep = 2.0 / float(max(uniforms.height - 1, 1));
    float slopeX = (surfaceHeight(column + 2, row) - surfaceHeight(column - 2, row)) /
                   (4.0 * columnStep);
    float slopeZ = (surfaceHeight(column, row + 2) - surfaceHeight(column, row - 2)) /
                   (4.0 * rowStep);

    vec3 position = vec3(float(column) * columnStep - 1.0,
                         elevation(value) * uniforms.heightScale,
                         float(row) * rowStep - 1.0);

    vec4 clip = uniforms.viewProjection * vec4(position, 1.0);
    gl_Position = clip;
    outMagnitude = value;
    outNormal = normalize(vec3(-slopeX, 1.0, -slopeZ));
    outAge = 1.0 - float(row) / float(max(uniforms.height - 1, 1));
}
