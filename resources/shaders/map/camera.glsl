#ifndef JETSTREAM_MAP_CAMERA_GLSL
#define JETSTREAM_MAP_CAMERA_GLSL

// Keep this std140 layout in sync with MapContext::GpuUniforms.
layout(set = 0, binding = 0) uniform ShaderUniforms {
    mat4 viewProjection;
    vec4 cameraPos;
    vec4 targetNormal;  // xyz = sub-camera normal, w = horizon threshold
    float surfaceScale;
    float viewportWidth;
    float viewportHeight;
    float lineWidth;
    float colorR;
    float colorG;
    float colorB;
    float lineStyle;
    float dashScale;
    float dashPhase;
    float outlineStrength;
    float lineOpacity;
    vec4 anchor;
} uniforms;

vec3 lonLatToSphere(float lon, float lat) {
    float r = radians(lat);
    float lr = radians(lon);
    float cl = cos(r);
    return vec3(cl * sin(lr), sin(r), cl * cos(lr));
}

vec3 mapLocal(vec2 lonLat) {
    float turns = round((lonLat.x + uniforms.anchor.z) / 360.0);
    vec2 delta = radians(vec2(lonLat.x - 360.0 * turns, lonLat.y) + uniforms.anchor.zw);
    vec2 base = radians(uniforms.anchor.xy);
    vec2 halfDelta = sin(delta * 0.5);
    vec2 middle = base + delta * 0.5;
    float sinLon = 2.0 * cos(middle.x) * halfDelta.x;
    float cosLon = -2.0 * sin(middle.x) * halfDelta.x;
    float sinLat = 2.0 * cos(middle.y) * halfDelta.y;
    float cosLat = -2.0 * sin(middle.y) * halfDelta.y;
    float latCos = cos(base.y) + cosLat;
    return vec3(latCos * sinLon + sin(base.x) * cosLat,
                sinLat,
                latCos * cosLon + cos(base.x) * cosLat);
}

vec3 mapSphere(vec2 lonLat) {
    vec2 absolute = lonLat + uniforms.anchor.zw + uniforms.anchor.xy;
    return lonLatToSphere(absolute.x, absolute.y);
}

vec3 sphereLocal(vec3 p) {
    return p - lonLatToSphere(uniforms.anchor.x, uniforms.anchor.y);
}

vec3 clippedLocal(vec3 clipped, vec3 original, vec2 lonLat) {
    return all(equal(clipped, original)) ? mapLocal(lonLat) : sphereLocal(clipped);
}

vec2 projectToNdc(vec3 p) {
    vec4 clip = uniforms.viewProjection * vec4(p, 1.0);
    return clip.xy / max(clip.w, 1e-7);
}

float tileFade(vec2 fade) {
    float span = max(uniforms.dashScale * 0.5, 1e-3);
    float now = uniforms.dashPhase;
    float fadeIn = smoothstep(fade.x, fade.x + span, now);
    float fadeOut = fade.y < 0.0 ? 1.0 : 1.0 - smoothstep(fade.y + span, fade.y + 2.0 * span, now);
    return fadeIn * fadeOut;
}

vec2 mapPixelSize() {
    return vec2(
        (2.0 * uniforms.surfaceScale) / max(uniforms.viewportWidth, 1.0),
        (2.0 * uniforms.surfaceScale) / max(uniforms.viewportHeight, 1.0)
    );
}

#endif
