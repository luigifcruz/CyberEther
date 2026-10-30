#ifndef JETSTREAM_RENDER_SAKURA_NODE_FLOAT_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_FLOAT_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct NodeFloatInput : public Component {
    struct Config {
        std::string id;
        F32 value = 0.0f;
        std::string unit;
        I32 precision = 2;
        std::optional<F32> step;
        Callback<void(F32)> onChange;
        Callback<void(F32)> onStepChange;

        bool operator==(const Config&) const = default;
    };

    NodeFloatInput();
    ~NodeFloatInput();

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

#endif  // JETSTREAM_RENDER_SAKURA_NODE_FLOAT_INPUT_HH
