#ifndef JETSTREAM_LPT_GENERIC_H
#define JETSTREAM_LPT_GENERIC_H

#include "jetstream/base.hpp"
#include "render/base.hpp"

namespace Jetstream::Lineplot {

struct Config {
    std::shared_ptr<Render::Instance> render;
    int width = 5000;
    int height = 1000;
};

class Generic : public Module {
public:
    explicit Generic(Config& c) : cfg(c) {};
    virtual ~Generic() = default;

    std::shared_ptr<Render::Texture> tex() const {
        return texture;
    };

    Config conf() const {
        return cfg;
    }

protected:
    Config& cfg;

    std::vector<float> a;
    std::vector<float> l;

    Render::Texture::Config textureCfg;
    Render::Program::Config programCfg;
    Render::Surface::Config surfaceCfg;
    Render::Vertex::Config gridVertexCfg;
    Render::Vertex::Config lineVertexCfg;
    Render::Draw::Config drawGridVertexCfg;
    Render::Draw::Config drawLineVertexCfg;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Vertex> lineVertex;
    std::shared_ptr<Render::Draw> drawGridVertex;
    std::shared_ptr<Render::Draw> drawLineVertex;

    const GLchar* vertexSource = R"END(#version 300 es
        layout (location = 0) in vec3 aPos;

        out vec2 TexCoord;

        void main() {
            gl_Position = vec4(aPos, 1.0);
        }
    )END";

    const GLchar* fragmentSource = R"END(#version 300 es
        precision highp float;

        out vec4 FragColor;

        uniform int drawIndex;

        void main() {
            if (drawIndex == 0) {
                FragColor = vec4(0.27, 0.27, 0.27, 0.0);
            } else if (drawIndex == 1) {
                FragColor = vec4(1.0, 0.0, 0.0, 0.0);
            } else {
                FragColor = vec4(0.0, 1.0, 0.0, 0.0);
            }
        }
    )END";
};

} // namespace Jetstream::Lineplot

#endif
