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
} uniforms;

vec3 lonLatToSphere(float lon, float lat) {
    float r = radians(lat);
    float lr = radians(lon);
    float cl = cos(r);
    return vec3(cl * sin(lr), sin(r), cl * cos(lr));
}

vec2 projectToNdc(vec3 p) {
    vec4 clip = uniforms.viewProjection * vec4(p, 1.0);
    return clip.xy / max(clip.w, 1e-7);
}

vec2 mapPixelSize() {
    return vec2(
        (2.0 * uniforms.surfaceScale) / max(uniforms.viewportWidth, 1.0),
        (2.0 * uniforms.surfaceScale) / max(uniforms.viewportHeight, 1.0)
    );
}

#endif
