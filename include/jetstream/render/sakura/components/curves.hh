#ifndef JETSTREAM_RENDER_SAKURA_CURVES_HH
#define JETSTREAM_RENDER_SAKURA_CURVES_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Curves : public Component {
    struct Instance {
        Extent2D<F32> p0 = {0.0f, 0.0f};
        Extent2D<F32> p1 = {0.0f, 0.0f};
        Extent2D<F32> p2 = {0.0f, 0.0f};
        Extent2D<F32> p3 = {0.0f, 0.0f};
        ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
        F32 thickness = 1.0f;
        U64 segments = 1;

        bool operator==(const Instance&) const = default;
    };

    struct Config {
        std::vector<Instance> instances;
        Extent2D<F32> origin = {0.0f, 0.0f};
        F32 scale = 1.0f;
        std::optional<Rect> clip;
        F32 fringe = 1.0f;
        U64 capacity = 1024;
        U64 maxSegments = 256;

        bool operator==(const Config&) const = default;
    };

    Curves();
    ~Curves();

    void update(Config config);
    const Config& config() const;

 protected:
    Result build(Context& ctx) override;
    Result paint() override;
    void conceal() override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CURVES_HH
