#include "jetstream/modules/constellation.hh"

#include "jetstream/render/components/shapes.hh"
#include "jetstream/render/utils.hh"
#include "jetstream/constants.hh"

#include "benchmark.cc"
#include "jetstream/types.hh"

namespace Jetstream {

template<Device D, typename T>
struct Constellation<D, T>::GImpl {
    std::shared_ptr<Render::Texture> framebufferTexture;
    std::shared_ptr<Render::Surface> surface;
    std::shared_ptr<Render::Components::Shapes> shapes;

    U64 numberOfPoints;

};

template<Device D, typename T>
Result Constellation<D, T>::create() {
    JST_DEBUG("Initializing Constellation module.");
    JST_INIT_IO();

    // Check input parameters.

    if (input.buffer.rank() == 0) {
        JST_ERROR("Input buffer rank is 0.");
        return Result::ERROR;
    }

    if (input.buffer.size() == 0) {
        JST_ERROR("Input is empty during initialization.");
        return Result::ERROR;
    }

    gimpl->numberOfPoints = input.buffer.size();

    return Result::SUCCESS;
}

template<Device D, typename T>
void Constellation<D, T>::info() const {
    JST_DEBUG("  Window Size: [{}, {}]", config.viewSize.x, config.viewSize.y);
}

template<Device D, typename T>
Result Constellation<D, T>::createPresent() {
    // Create shapes component.
    {
        Render::Components::Shapes::Config cfg;
        cfg.pixelSize = {2.0f / config.viewSize.x, 2.0f / config.viewSize.y};
        cfg.elements["constellation_points"] = {
            .type = Render::Components::Shapes::Type::CIRCLE,
            .numberOfInstances = gimpl->numberOfPoints,
            .size = {10.0f, 10.0f},
        };

        JST_CHECK(window->build(gimpl->shapes, cfg));
        JST_CHECK(window->bind(gimpl->shapes));
    }

    // Create framebuffer texture.
    {
        Render::Texture::Config cfg;
        cfg.size = config.viewSize;
        JST_CHECK(window->build(gimpl->framebufferTexture, cfg));
    }

    // Create surface.
    {
        Render::Surface::Config cfg;
        cfg.framebuffer = gimpl->framebufferTexture;
        cfg.clearColor = {0.1f, 0.1f, 0.1f, 1.0f};
        JST_CHECK(gimpl->shapes->surface(cfg));
        JST_CHECK(window->build(gimpl->surface, cfg));
        JST_CHECK(window->bind(gimpl->surface));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::present() {
    // Update pixel size in case view size changed.
    gimpl->shapes->updatePixelSize({2.0f / config.viewSize.x, 2.0f / config.viewSize.y});

    // Present the shapes
    JST_CHECK(gimpl->shapes->present());

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::destroyPresent() {
    JST_CHECK(window->unbind(gimpl->surface));
    JST_CHECK(window->unbind(gimpl->shapes));

    return Result::SUCCESS;
}

template<Device D, typename T>
const Extent2D<U64>& Constellation<D, T>::viewSize(const Extent2D<U64>& viewSize) {
    if (gimpl->surface->size(viewSize) != this->viewSize()) {
        JST_TRACE("Constellation size changed from [{}, {}] to [{}, {}].",
                config.viewSize.x,
                config.viewSize.y,
                viewSize.x,
                viewSize.y);

        config.viewSize = gimpl->surface->size();
    }
    return this->viewSize();
}

template<Device D, typename T>
Render::Texture& Constellation<D, T>::getTexture() {
    return *gimpl->framebufferTexture;
};

}  // namespace Jetstream
