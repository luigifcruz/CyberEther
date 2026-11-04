#ifndef JETSTREAM_RENDER_SAKURA_DIVIDER_HH
#define JETSTREAM_RENDER_SAKURA_DIVIDER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/types.hh>

#include <memory>

namespace Jetstream::Sakura {

struct Divider : public Component {
    struct Config {
        F32 spacing = Stack::Spacing;
        bool separator = true;

        bool operator==(const Config&) const = default;
    };

    Divider();
    ~Divider();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DIVIDER_HH
