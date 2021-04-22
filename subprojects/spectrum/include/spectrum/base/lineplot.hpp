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

    const GLchar* vertexSource = R"END(#version 300 es
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec2 TexCoord;

        void main() {
            gl_Position = vec4(aPos, 1.0);
            TexCoord = aTexCoord;
        }
    )END";

    const GLchar* fragmentSource = R"END(#version 300 es
        precision highp float;

        out vec4 FragColor;

        in vec2 TexCoord;

        uniform float Scale;
        uniform int vertexIdx;
        uniform sampler2D ourTexture2;

        void main() {
            if (vertexIdx == 1) {
                FragColor = vec4(Scale, 1.0, 0.0, 0.0);
            } else {
                FragColor = texture(ourTexture2, TexCoord);
            }
        }
    )END";
};

} // namespace Render

#endif

