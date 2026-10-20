#ifndef JETSTREAM_RENDER_SAKURA_PILL_STEPPER_HH
#define JETSTREAM_RENDER_SAKURA_PILL_STEPPER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct PillStepper : public Component {
    struct Config {
        std::string label;
        bool canDecrement = true;
        bool canIncrement = true;
        Extent2D<F32> size = {0.0f, 34.0f};
        Callback<void()> onDecrement;
        Callback<void()> onIncrement;
        Callback<void()> onReset;

        bool operator==(const Config&) const = default;
    };

    PillStepper();
    ~PillStepper();

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

#endif  // JETSTREAM_RENDER_SAKURA_PILL_STEPPER_HH
