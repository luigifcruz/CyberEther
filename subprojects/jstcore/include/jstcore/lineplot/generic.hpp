#ifndef JSTCORE_LINEPLOT_GENERIC_H
#define JSTCORE_LINEPLOT_GENERIC_H

#include "jetstream/base.hpp"
#include "render/tools/lut.hpp"
#include "render/base.hpp"

namespace Jetstream::Lineplot {

class Generic : public Module {
public:
    struct Config {
        Size2D<int> size {3072, 500};
    };

    struct Input {
        const Data<VF32> in;
    };

    explicit Generic(const Config&, const Input&);
    virtual ~Generic() = default;

    Result compute();
    Result present();

    constexpr Size2D<int> size() const {
        return config.size;
    }
    Size2D<int> size(const Size2D<int>&);

    Render::Texture& tex() const;

protected:
    Config config;
    const Input input;

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

    Result initRender(float*, bool cudaInterop = false);
};

} // namespace Jetstream::Lineplot

#endif
