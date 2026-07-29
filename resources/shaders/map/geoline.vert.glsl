#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) uniform ShaderUniforms {
    float centerLon;
    float centerLat;
    float zoom;
    float aspectRatio;
    float surfaceScale;
    float lineWidth;
    float colorR;
    float colorG;
    float colorB;
    float viewportWidth;
    float viewportHeight;
} uniforms;

// Quad vertex: x = endpoint selector (0 or 1), y = side (-1 or +1).
layout(location = 0) in vec2 inQuad;

// Instance data: (x1, y1, x2, y2) in Mercator space.
layout(location = 1) in vec4 inSegment;

layout(location = 0) out vec2 vNormal;

const float PI = 3.14159265358979323846;
const float MAX_MERCATOR_LAT = 85.05112878;

float mercatorX(float lon) {
    return (lon + 180.0) / 360.0;
}

float mercatorY(float lat) {
    lat = clamp(lat, -MAX_MERCATOR_LAT, MAX_MERCATOR_LAT);
    float r = radians(lat);
    return (1.0 - asinh(tan(r)) / PI) / 2.0;
}

float wrapMercatorDelta(float delta) {
    if (delta > 0.5) {
        return delta - 1.0;
    }
    if (delta < -0.5) {
        return delta + 1.0;
    }
    return delta;
}

void main() {
    vec2 a = inSegment.xy;
    vec2 b = inSegment.zw;

    float cx = mercatorX(uniforms.centerLon);
    float cy = mercatorY(uniforms.centerLat);
    float scale = pow(2.0, uniforms.zoom);

    float ax = wrapMercatorDelta(a.x - cx);
    float bx = wrapMercatorDelta(b.x - cx);
    if (bx - ax > 0.5) {
        bx -= 1.0;
    } else if (bx - ax < -0.5) {
        bx += 1.0;
    }

    vec2 start = vec2(ax * scale * 2.0 / uniforms.aspectRatio,
                      (cy - a.y) * scale * 2.0);
    vec2 end = vec2(bx * scale * 2.0 / uniforms.aspectRatio,
                    (cy - b.y) * scale * 2.0);
    vec2 current = mix(start, end, inQuad.x);

    vec2 direction = end - start;
    float segmentLength = length(direction);
    if (segmentLength < 1e-7) {
        gl_Position = vec4(current, 0.0, 1.0);
        vNormal = vec2(0.0);
        return;
    }

    direction /= segmentLength;
    vec2 perpendicular = vec2(-direction.y, direction.x);
    vec2 pixelToNdc = vec2(
        (2.0 * uniforms.surfaceScale) / uniforms.viewportWidth,
        (2.0 * uniforms.surfaceScale) / uniforms.viewportHeight
    );
    vec2 thickness = vec2(uniforms.lineWidth * 0.5) * pixelToNdc;
    vec2 offset = perpendicular * thickness * inQuad.y;

    gl_Position = vec4(current + offset, 0.0, 1.0);
    vNormal = vec2(0.0, (inQuad.y + 1.0) * 0.5);
}
