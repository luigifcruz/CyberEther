#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inLocalPos;
layout(location = 1) in vec4 inFillColor;
layout(location = 2) in vec4 inBorderColor;
layout(location = 3) in vec4 inShapeParams; // x=type, y=borderWidth, z=cornerRadius, w=corner mask (0 = all)
layout(location = 4) in vec2 inPixelSize;

layout(location = 0) out vec4 outColor;

// Shape type constants
const int TYPE_TRIANGLE = 0;
const int TYPE_RECT = 1;
const int TYPE_CIRCLE = 2;

float sdfCircle(vec2 p, float radius) {
    return length(p) - radius;
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

float alphaRect(vec2 p, vec2 size, vec2 localPixel) {
    // Use separable box coverage to keep thin rectangles crisp.
    vec2 halfSize = size * 0.5;
    vec2 insideDistance = halfSize - abs(p);

    vec2 coverage = clamp(insideDistance / localPixel + 0.5, 0.0, 1.0);
    return coverage.x * coverage.y;
}

float roundedRectSdPixels(vec2 p, float radiusPixels, vec2 localPixel, float insetPixels) {
    vec2 halfSizePixels = max(vec2(0.5) / localPixel - vec2(insetPixels), vec2(0.0));
    float radius = clamp(radiusPixels, 0.0, min(halfSizePixels.x, halfSizePixels.y));

    vec2 pPixels = p / localPixel;
    vec2 q = abs(pPixels) - (halfSizePixels - vec2(radius));
    return length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

void main() {
    int shapeType = int(inShapeParams.x);
    float borderWidth = inShapeParams.y;
    int params = int(inShapeParams.w + 0.5);
    int corners = params & 15;
    bool strokeFringe = (params & 16) != 0;
    bool soft = (params & 32) != 0;
    bool topLit = (params & 64) != 0;
    if (corners == 0) {
        corners = 15;
    }
    int corner = inLocalPos.y > 0.0 ? (inLocalPos.x > 0.0 ? 2 : 1) : (inLocalPos.x > 0.0 ? 4 : 8);
    float cornerRadius = (corners & corner) != 0 ? inShapeParams.z : 0.0;

    // WGSL requires derivatives in uniform control flow, so compute every
    // pixel-scale term up front before branching on per-instance params.
    vec2 localPixel = max(vec2(length(vec2(dFdx(inLocalPos.x), dFdy(inLocalPos.x))),
                               length(vec2(dFdx(inLocalPos.y), dFdy(inLocalPos.y)))),
                          vec2(1.0e-6));

    float circleSd = sdfCircle(inLocalPos, 0.5);
    float circleSdPixels = circleSd / max(length(vec2(dFdx(circleSd), dFdy(circleSd))), 1.0e-6);

    float triangleSd = sdfTriangle(inLocalPos);
    float triangleSdPixels = triangleSd / max(length(vec2(dFdx(triangleSd), dFdy(triangleSd))), 1.0e-6);

    float rectSdPixels = roundedRectSdPixels(inLocalPos, cornerRadius, localPixel, 0.0);
    float softness = max(borderWidth, 1.0);
    float softSdPixels = roundedRectSdPixels(inLocalPos, cornerRadius, localPixel, softness);
    float topPixels = (0.5 - inLocalPos.y) / localPixel.y;
    float topFade = 1.0 - smoothstep(0.0, max(2.0 * inShapeParams.z, 1.0), topPixels);

    if (soft) {
        float softAlpha = 1.0 - smoothstep(-softness, softness, softSdPixels);
        outColor = vec4(inFillColor.rgb, inFillColor.a * softAlpha);
        return;
    }

    float sdPixels;
    if (shapeType == TYPE_RECT) {
        sdPixels = rectSdPixels;
    } else if (shapeType == TYPE_CIRCLE) {
        sdPixels = circleSdPixels;
    } else { // TYPE_TRIANGLE
        sdPixels = triangleSdPixels;
    }

    if (borderWidth <= 0.0) {
        float fillAlpha = (shapeType == TYPE_RECT && cornerRadius <= 0.0)
            ? alphaRect(inLocalPos, vec2(1.0), localPixel)
            : clamp(0.5 - sdPixels, 0.0, 1.0);
        outColor = vec4(inFillColor.rgb, inFillColor.a * fillAlpha);
        return;
    }

    float outer = clamp(0.5 - sdPixels, 0.0, 1.0);

    if (strokeFringe) {
        float stroke = clamp(1.0 - abs(sdPixels + 0.5 * borderWidth) / borderWidth, 0.0, 1.0) * inBorderColor.a *
                       (topLit ? topFade : 1.0);
        vec4 fill = vec4(inFillColor.rgb * inFillColor.a, inFillColor.a);
        vec4 composed = vec4(inBorderColor.rgb * stroke, stroke) + fill * (1.0 - stroke);
        outColor = vec4(composed.rgb / max(composed.a, 1.0e-6), composed.a * outer);
        return;
    }

    float inner = clamp(0.5 - (sdPixels + borderWidth), 0.0, 1.0);

    vec3 rgb = mix(inBorderColor.rgb, inFillColor.rgb, inner);
    float alpha = outer * mix(inBorderColor.a, inFillColor.a, inner);
    outColor = vec4(rgb, alpha);
}
