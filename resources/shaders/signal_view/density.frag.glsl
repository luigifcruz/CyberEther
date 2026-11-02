#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ShaderUniforms {
    uint columns;
    uint levels;
    float binsPerColumn;
    float binSpan;
    float offset;
    float zoom;
    float panelScaleX;
    float panelScaleY;
    float plotWidth;
    float intensityScale;
    float panelOffsetY;
} uniforms;

layout(set = 0, binding = 1) readonly buffer DensityBuffer {
    float density[];
};

layout(set = 0, binding = 2) uniform texture2D lutTex;
layout(set = 0, binding = 3) uniform sampler lutSam;

const float kResponse = 90.0;

float cell(int column, int levelPosition) {
    if (column < 0 || column >= int(uniforms.columns)) {
        return 0.0;
    }
    int levels = int(uniforms.levels);
    return density[column * levels + clamp(levelPosition, 0, levels - 1)];
}

float levelSample(int column, float levelPosition) {
    float position = levelPosition - 0.5;
    float base = floor(position);
    return mix(cell(column, int(base)), cell(column, int(base) + 1), position - base);
}

float bilinearSample(float column, float levelPosition) {
    float position = column - 0.5;
    float base = floor(position);
    return mix(levelSample(int(base), levelPosition), levelSample(int(base) + 1, levelPosition), position - base);
}

void main() {
    float columnsPerPixel = float(uniforms.columns) /
                            max(uniforms.zoom * uniforms.plotWidth, 1.0);
    int columnTaps = clamp(int(ceil(columnsPerPixel)), 1, 16);
    float columnFootprint = max(columnsPerPixel, 1.0);

    float hits = 0.0;
    for (int i = 0; i < columnTaps; ++i) {
        float column = inTexcoord.x;
        if (columnTaps > 1) {
            column += ((float(i) + 0.5) / float(columnTaps) - 0.5) * columnFootprint;
        }
        hits += bilinearSample(column, inTexcoord.y);
    }
    hits /= float(columnTaps);

    float occupancy = hits * uniforms.intensityScale;
    float intensity = 1.0 - exp(-kResponse * occupancy);
    vec3 color = textureLod(sampler2D(lutTex, lutSam), vec2(clamp(intensity, 0.0, 1.0), 0.0), 0.0).rgb;
    outColor = vec4(color, 1.0);
}
