#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in float inElevation;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in float inAge;

layout(location = 0) out vec4 outColor;

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

layout(set = 0, binding = 2) uniform texture2D lutTex;
layout(set = 0, binding = 3) uniform sampler lutSam;

void main() {
    float mapped = 0.5 + 0.5 * (2.0 * inElevation - 1.0) * tanh(2.0);
    vec3 base = texture(sampler2D(lutTex, lutSam), vec2(mapped, 0.0)).rgb;

    float diffuse = max(dot(normalize(inNormal), normalize(uniforms.lightDirection.xyz)), 0.0);
    float ambient = uniforms.lightDirection.w;
    vec3 shaded = base * (ambient + (1.0 - ambient) * diffuse);

    float fade = pow(clamp(inAge, 0.0, 1.0), uniforms.fade.y) * uniforms.fade.x;
    vec3 color = mix(shaded, uniforms.background.rgb, fade);

    outColor = vec4(color, 1.0);
}
