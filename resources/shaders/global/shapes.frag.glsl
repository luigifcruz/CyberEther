#version 450
#extension GL_ARB_separate_shader_objects : enable

// TODO: Implement borders.
// TODO: Implement rounded corners.
// TODO: Implement propper lines.

layout(location = 0) in vec2 inLocalPos;
layout(location = 1) in vec4 inFillColor;
layout(location = 2) in vec4 inBorderColor;
layout(location = 3) in vec4 inShapeParams; // x=type, y=borderWidth, z=cornerRadius, w=unused
layout(location = 4) in vec2 inPixelSize;

layout(location = 0) out vec4 outColor;

// Shape type constants
const int TYPE_TRIANGLE = 0;
const int TYPE_RECT = 1;
const int TYPE_CIRCLE = 2;
const int TYPE_LINE = 3;

float sdfCircle(vec2 p, float radius) {
    return length(p) - radius;
}

float sdfRect(vec2 p, vec2 size) {
    vec2 d = abs(p) - size * 0.5;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float sdfTriangle(vec2 p) {
    // Optimized equilateral triangle for better antialiasing
    const float k = sqrt(3.0);
    p.x = abs(p.x) - 0.5;
    p.y = p.y + 1.0 / sqrt(3.0);
    if (p.x + k * p.y > 0.0) {
        p = vec2(p.x - k * p.y, -k * p.x - p.y) * 0.5;
    }
    p.x -= clamp(p.x, -2.0 / sqrt(3.0), 2.0 / sqrt(3.0));
    return -length(p) * sign(p.y);
}

void main() {
    int shapeType = int(inShapeParams.x);
    float borderWidth = inShapeParams.y;
    float cornerRadius = inShapeParams.z;

    float distance = 0.0;

    // Calculate signed distance based on shape type
    if (shapeType == TYPE_CIRCLE) {
        distance = sdfCircle(inLocalPos, 0.5);
    } else if (shapeType == TYPE_RECT || shapeType == TYPE_LINE) {
        distance = sdfRect(inLocalPos, vec2(1.0));
    } else if (shapeType == TYPE_TRIANGLE) {
        distance = sdfTriangle(inLocalPos);
    }

    // Calculate the gradient of the distance field for antialiasing
    float width = fwidth(distance);

    // Edge sharpness control
    float edgeSharpness = 0.75;

    // Shape fill with improved antialiasing
    float fillAlpha = smoothstep(edgeSharpness * width, -edgeSharpness * width, distance);

    // Combine fill and border
    outColor = vec4(inFillColor.r, inFillColor.g, inFillColor.b, fillAlpha);
}
