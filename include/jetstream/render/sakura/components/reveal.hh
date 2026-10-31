#ifndef JETSTREAM_RENDER_SAKURA_REVEAL_HH
#define JETSTREAM_RENDER_SAKURA_REVEAL_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>

namespace Jetstream::Sakura {

struct Reveal : public Component {
    struct Config {
        Component* child = nullptr;
        bool shown = false;
        bool upward = false;
        F32 gap = 0.0f;
        F64 minimumDuration = 0.0;
        F64 linger = 0.0;

        bool operator==(const Config&) const = default;
    };

    Reveal();
    ~Reveal();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_REVEAL_HH
