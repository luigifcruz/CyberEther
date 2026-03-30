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

// Instance data: (lon1, lat1, lon2, lat2) per segment.
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
    return (1.0 - log(tan(r) + 1.0 / cos(r)) / PI) / 2.0;
}

void main() {
    // Select endpoint based on quad x.
    float lon = mix(inSegment.x, inSegment.z, inQuad.x);
    float lat = mix(inSegment.y, inSegment.w, inQuad.x);

    // Project both endpoints through Web Mercator.
    float cx = mercatorX(uniforms.centerLon);
    float cy = mercatorY(uniforms.centerLat);
    float scale = pow(2.0, uniforms.zoom);

    // Current vertex position in NDC.
    float vx = (mercatorX(lon) - cx) * scale * 2.0;
    float vy = (cy - mercatorY(lat)) * scale * 2.0;
    vx /= uniforms.aspectRatio;

    // Both endpoints in NDC (needed for perpendicular direction).
    float ax = (mercatorX(inSegment.x) - cx) * scale * 2.0;
    float ay = (cy - mercatorY(inSegment.y)) * scale * 2.0;
    ax /= uniforms.aspectRatio;

    float bx = (mercatorX(inSegment.z) - cx) * scale * 2.0;
    float by = (cy - mercatorY(inSegment.w)) * scale * 2.0;
    bx /= uniforms.aspectRatio;

    // Direction along the line in NDC.
    vec2 dir = vec2(bx - ax, by - ay);
    float len = length(dir);

    // Degenerate segment guard.
    if (len < 1e-7) {
        gl_Position = vec4(vx, vy, 0.0, 1.0);
        vNormal = vec2(0.0, 0.0);
        return;
    }

    dir /= len;

    // Perpendicular in NDC.
    vec2 perp = vec2(-dir.y, dir.x);

    // Convert thickness from pixels to NDC — same as thicklines.
    vec2 pixToNDC = vec2((2.0 * uniforms.surfaceScale) / uniforms.viewportWidth,
                         (2.0 * uniforms.surfaceScale) / uniforms.viewportHeight);
    vec2 thickness = vec2(uniforms.lineWidth * 0.5) * pixToNDC;
    vec2 offset = perp * thickness * inQuad.y;

    gl_Position = vec4(vx + offset.x, vy + offset.y, 0.0, 1.0);

    // Normal for AA: same encoding as thicklines.
    // inQuad.y = -1 → normal = (0, 0), inQuad.y = +1 → normal = (0, 1).
    vNormal = vec2(0.0, (inQuad.y + 1.0) * 0.5);
}
