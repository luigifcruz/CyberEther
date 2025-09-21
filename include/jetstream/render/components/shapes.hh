#ifndef JETSTREAM_RENDER_COMPONENTS_SHAPES_HH
#define JETSTREAM_RENDER_COMPONENTS_SHAPES_HH

#include <span>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/components/generic.hh"

namespace Jetstream::Render::Components {

// TODO: Add support for zero-copy buffers.
// TODO: Implement proper rotation.

class Shapes : public Generic {
 public:
    enum class Type : uint32_t {
        TRIANGLE = 0,
        RECT = 1,
        CIRCLE = 2,
    };

    struct ElementConfig {
        Type type = Type::RECT;
        U64 numberOfInstances = 1;
        ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
        F32 rotation = 0.0f;
        Extent2D<F32> position = {0.0f, 0.0f};
        Extent2D<F32> size = {0.0f, 0.0f};
        F32 cornerRadius = 0.0f;
        F32 borderWidth = 0.0f;
        ColorRGBA<F32> borderColor = {0.0f, 0.0f, 0.0f, 1.0f};
    };

    struct Config {
        Extent2D<F32> pixelSize = {0.0f, 0.0f};
        std::unordered_map<std::string, ElementConfig> elements;
    };

    Shapes(const Config& config);
    ~Shapes();

    Result create(Window* window);
    Result destroy(Window* window);

    Result surface(Render::Surface::Config& config);

    Result present();

    Result getColors(const std::string& elementId, std::span<ColorRGBA<F32>>& colors) const;
    Result updateColors(const std::string& elementId = {});

    Result getRotations(const std::string& elementId, std::span<F32>& rotations) const;
    Result updateRotations(const std::string& elementId = {});

    Result getPositions(const std::string& elementId, std::span<Extent2D<F32>>& positions) const;
    Result updatePositions(const std::string& elementId = {});

    Result getSizes(const std::string& elementId, std::span<Extent2D<F32>>& sizes) const;
    Result updateSizes(const std::string& elementId = {});

    Result updatePixelSize(const Extent2D<F32>& pixelSize);

    constexpr const Config& getConfig() const {
        return config;
    }

 private:
    Config config;

    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Render::Components

#endif
