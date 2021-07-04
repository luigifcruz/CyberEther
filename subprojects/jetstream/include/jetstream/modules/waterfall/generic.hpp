#ifndef JETSTREAM_WTF_GENERIC_H
#define JETSTREAM_WTF_GENERIC_H

#include "jetstream/modules/module.hpp"
#include "jetstream/tools/lut.hpp"
#include "render/base.hpp"
#include "render/extras.hpp"

namespace Jetstream {

class Waterfall : public Module {
public:
    using TI = VF32;

    struct Config {
        std::shared_ptr<Render::Instance> render;
        bool interpolate = true;
        Size2D<int> size = {2500, 500};
    };

    static Connections inputBlueprint(const Locale & device) {
        switch (device) {
            case Locale::CPU:
            return {
                {"input0", Data<TI>{Locale::CPU, {}}},
            };
            case Locale::CUDA:
            return {
                {"input0", Data<TI>{Locale::CUDA, {}}},
            };
        }
        return {};
    }

    class CPU;
#ifdef JETSTREAM_WTF_CUDA_AVAILABLE
    class CUDA;
#endif

    explicit Waterfall(const Config & cfg, Connections & input);
    virtual ~Waterfall() = default;

    constexpr bool interpolate() const {
        return cfg.interpolate;
    }
    bool interpolate(bool);

    constexpr Size2D<int> size() const {
        return cfg.size;
    }
    Size2D<int> size(const Size2D<int> &);

    std::weak_ptr<Render::Texture> tex() const;

protected:
    Config cfg;
    Data<TI> in;

    int inc = 0, last = 0, ymax = 0;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Texture> binTexture;
    std::shared_ptr<Render::Texture> lutTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

    Result _initRender(uint8_t*, bool cudaInterop = false);

    virtual Result _compute() = 0;

    Result present() final;
    Result compute() final;

    const GLchar* vertexSource = R"END(#version 300 es
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        uniform float Index;

        void main() {
            gl_Position = vec4(aPos, 1.0);
            float coord = (Index-aTexCoord.y);
            TexCoord = vec2(aTexCoord.x, coord);
        }
    )END";

    const GLchar* fragmentSource = R"END(#version 300 es
        precision highp float;

        out vec4 FragColor;

        in vec2 TexCoord;

        uniform int Interpolate;
        uniform int drawIndex;
        uniform sampler2D BinTexture;
        uniform sampler2D LutTexture;

        vec4 cubic(float v){
            vec4 n = vec4(1.0, 2.0, 3.0, 4.0) - v;
            vec4 s = n * n * n;
            float x = s.x;
            float y = s.y - 4.0 * s.x;
            float z = s.z - 4.0 * s.y + 6.0 * s.x;
            float w = 6.0 - x - y - z;
            return vec4(x, y, z, w) * (1.0/6.0);
        }

        vec4 textureBicubic(sampler2D sampler, vec2 texCoords){
            vec2 texSize = vec2(textureSize(sampler, 0));
            vec2 invTexSize = 1.0 / texSize;

            texCoords = texCoords * texSize - 0.5;

            vec2 fxy = fract(texCoords);
            texCoords -= fxy;

            vec4 xcubic = cubic(fxy.x);
            vec4 ycubic = cubic(fxy.y);

            vec4 c = texCoords.xxyy + vec2 (-0.5, +1.5).xyxy;

            vec4 s = vec4(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
            vec4 offset = c + vec4 (xcubic.yw, ycubic.yw) / s;

            offset *= invTexSize.xxyy;

            vec4 sample0 = texture(sampler, offset.xz);
            vec4 sample1 = texture(sampler, offset.yz);
            vec4 sample2 = texture(sampler, offset.xw);
            vec4 sample3 = texture(sampler, offset.yw);

            float sx = s.x / (s.x + s.y);
            float sy = s.z / (s.z + s.w);

            return mix(mix(sample3, sample2, sx), mix(sample1, sample0, sx), sy);
        }

        void main() {
            float mag;

            if (Interpolate == 1) {
                mag = textureBicubic(BinTexture, TexCoord).r;
            }

            if (Interpolate == 0) {
                mag = texture(BinTexture, TexCoord).r;
            }

            FragColor = texture(LutTexture, vec2(mag, 0));
        }
    )END";
};

} // namespace Jetstream

#endif
