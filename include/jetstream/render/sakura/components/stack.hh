#ifndef JETSTREAM_RENDER_SAKURA_STACK_HH
#define JETSTREAM_RENDER_SAKURA_STACK_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <vector>

namespace Jetstream::Sakura {

struct Stack : public Component {
    static constexpr F32 Spacing = 8.0f;

    enum class Axis {
        Vertical,
        Horizontal,
    };

    struct Item {
        Component* child = nullptr;
        F32 size = 0.0f;
        F32 grow = 0.0f;
        F32 shrink = 0.0f;

        bool operator==(const Item&) const = default;
    };

    struct Config {
        Axis axis = Axis::Vertical;
        F32 spacing = Spacing;
        bool centerCross = true;
        Component* widthOf = nullptr;
        std::vector<Item> items;

        bool operator==(const Config&) const = default;
    };

    Stack();
    ~Stack();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_STACK_HH
