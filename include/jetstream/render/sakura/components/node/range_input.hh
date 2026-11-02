#ifndef JETSTREAM_RENDER_SAKURA_NODE_RANGE_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_RANGE_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct NodeRangeInput : public Component {
    struct Config {
        std::string id;
        F32 min = 0.0f;
        F32 max = 1.0f;
        F32 value = 0.0f;
        bool integer = false;
        std::string unit;
        I32 precision = 0;
        Callback<void(F32)> onChange;

        bool operator==(const Config&) const = default;
    };

    NodeRangeInput();
    ~NodeRangeInput();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_RANGE_INPUT_HH
