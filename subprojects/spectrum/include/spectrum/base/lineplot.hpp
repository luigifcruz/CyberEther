#ifndef SPECTRUM_BASE_LINEPLOT_H
#define SPECTRUM_BASE_LINEPLOT_H

#include "spectrum/types.hpp"

namespace Spectrum {

class LinePlot {
public:
    struct Config {
        int width = 1280;
        int height = 480;
    };

    LinePlot(Config& c) : cfg(c) {};
    virtual ~LinePlot() = default;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
    virtual Result draw() = 0;

    Config& config() {
        return cfg;
    }

    virtual uint raw() = 0;

protected:
    Config& cfg;

    std::vector<float> l;

    Render::Texture::Config textureCfg;
    Render::Program::Config programCfg;
    Render::Surface::Config surfaceCfg;
    Render::Vertex::Config gridVertexCfg;
    Render::Vertex::Config lineVertexCfg;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> gridVertex;
    std::shared_ptr<Render::Vertex> lineVertex;

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

        uniform int vertexIdx;

        void main() {
            if (vertexIdx == 0) {
                FragColor = vec4(0.27, 0.27, 0.27, 0.0);
            } else {
                FragColor = vec4(1.0, 0.0, 0.0, 0.0);
            }
        }
    )END";
};

} // namespace Render

#endif

