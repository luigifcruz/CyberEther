#ifndef JETSTREAM_LPT_GENERIC_H
#define JETSTREAM_LPT_GENERIC_H

#include "jetstream/base.hpp"
#include "jetstream/tools/lut.hpp"
#include "render/base.hpp"

namespace Jetstream {
namespace Lineplot {

using TI = nonstd::span<float>;

struct Config {
    Size2D<int> size {2500, 500};

    Data<TI> input0;
    Jetstream::Policy policy;
    std::shared_ptr<Render::Instance> render;
};

class Generic : public Module {
public:
    explicit Generic(const Config &);
    virtual ~Generic() = default;

    constexpr Size2D<int> size() const {
        return cfg.size;
    }
    Size2D<int> size(const Size2D<int> &);

    std::weak_ptr<Render::Texture> tex() const;

protected:
    Config cfg;
    Data<TI> in;

    std::vector<float> plot;
    std::vector<float> grid;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Texture> lutTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Vertex> lineVertex;
    std::shared_ptr<Render::Draw> drawGridVertex;
    std::shared_ptr<Render::Draw> drawLineVertex;

    Result _initRender(float*, bool cudaInterop = false);

    virtual Result _compute() = 0;
    virtual Result _present() = 0;

    Result underlyingPresent() final;
    Result underlyingCompute() final;

    const GLchar* vertexSource = R"END(#version 300 es
        layout (location = 0) in vec3 aPos;

        out vec2 A;

        void main() {
            float min_x = 0.0;
            float max_x = 1.0;
            float y = -(2.0 * ((aPos.y - min_x)/(max_x - min_x)) - 1.0);
            gl_Position = vec4(aPos.x, y, aPos.z, 1.0);
            float pos = (y + 1.0)/2.0;
            A = vec2(-pos, 0.0);
        }
    )END";

    const GLchar* fragmentSource = R"END(#version 300 es
        precision highp float;

        out vec4 FragColor;

        in vec2 A;

        uniform int drawIndex;
        uniform sampler2D LutTexture;

        void main() {
            if (drawIndex == 0) {
                FragColor = vec4(0.27, 0.27, 0.27, 0.0);
            } else if (drawIndex == 1) {
                FragColor = texture(LutTexture, A);
            } else {
                FragColor = vec4(0.0, 1.0, 0.0, 0.0);
            }
        }
    )END";
};

} // namespace Lineplot
} // namespace Jetstream

#endif
