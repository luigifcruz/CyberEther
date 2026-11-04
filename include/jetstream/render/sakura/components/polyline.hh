#ifndef JETSTREAM_RENDER_SAKURA_POLYLINE_HH
#define JETSTREAM_RENDER_SAKURA_POLYLINE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Polyline : public Component {
    struct Instance {
        std::vector<Extent2D<F32>> points;
        ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
        F32 thickness = 1.0f;
        bool closed = false;
        bool filled = false;
        bool visible = true;

        bool operator==(const Instance&) const = default;
    };

    struct Config {
        std::vector<Instance> instances;
        std::optional<Rect> clip;
        Extent2D<F32> origin = {0.0f, 0.0f};
        F32 scale = 1.0f;
        F32 fringe = 1.0f;
        U64 capacity = 4096;

        bool operator==(const Config&) const = default;
    };


    static std::vector<Extent2D<F32>> Bezier(Extent2D<F32> p0,
                                             Extent2D<F32> p1,
                                             Extent2D<F32> p2,
                                             Extent2D<F32> p3,
                                             U64 segments);
    static std::vector<Extent2D<F32>> Circle(Extent2D<F32> center, F32 radius, U64 segments);
    static std::vector<Instance> CornerMasks(Rect rect,
                                             F32 radius,
                                             ColorRGBA<F32> top,
                                             ColorRGBA<F32> bottom,
                                             U64 segments = 8);

    Polyline();
    ~Polyline();

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

#endif  // JETSTREAM_RENDER_SAKURA_POLYLINE_HH
