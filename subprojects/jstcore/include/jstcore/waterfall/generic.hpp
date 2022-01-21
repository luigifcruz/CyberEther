#ifndef JSTCORE_WATERFALL_GENERIC_H
#define JSTCORE_WATERFALL_GENERIC_H

#include "jetstream/base.hpp"
#include "render/tools/lut.hpp"
#include "render/base.hpp"
#include "render/extras.hpp"

namespace Jetstream::Waterfall {

class Generic : public Module {
public:
    struct Config {
        bool interpolate = true;
        Size2D<int> size = {2500, 2000};
        float zoom = 1.0;
        int offset = 0.0;
    };

    struct Input {
        const Data<VF32> in;
    };

    explicit Generic(const Config&, const Input&);
    virtual ~Generic() = default;

    Result compute();
    Result present();

    constexpr bool interpolate() const {
        return config.interpolate;
    }
    bool interpolate(bool);

    constexpr Size2D<int> size() const {
        return config.size;
    }
    Size2D<int> size(const Size2D<int>&);

    constexpr float zoom() const {
        return config.zoom;
    }
    float zoom(const float&);

    constexpr int offset() const {
        return config.offset;
    }
    int offset(const int&);

    Render::Texture& tex() const;

protected:
    Config config;
    const Input input;

    int inc = 0, last = 0, ymax = 0;

    struct {
        int width;
        int height;
        int maxSize;
        float index;
        float offset;
        float zoom;
        bool interpolate;
    } shaderUniforms;

    std::shared_ptr<Render::Buffer> fillScreenVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenTextureVerticesBuffer;
    std::shared_ptr<Render::Buffer> fillScreenIndicesBuffer;

    std::shared_ptr<Render::Texture> texture;
    std::shared_ptr<Render::Buffer> binTexture;
    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Texture> lutTexture;
    std::shared_ptr<Render::Program> program;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> drawVertex;

    Result initRender(uint8_t*, bool cudaInterop = false);
};

} // namespace Jetstream::Waterfall

#endif
