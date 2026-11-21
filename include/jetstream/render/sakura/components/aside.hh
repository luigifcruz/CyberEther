#ifndef JETSTREAM_RENDER_SAKURA_ASIDE_HH
#define JETSTREAM_RENDER_SAKURA_ASIDE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>

namespace Jetstream::Sakura {

struct Aside : public Component {
    struct Config {
        Component* content = nullptr;
        Component* rail = nullptr;
        F32 railWidth = 260.0f;
        F32 railInset = 0.0f;
        F32 minimumContentWidth = 640.0f;

        bool operator==(const Config&) const = default;
    };

    Aside();
    ~Aside();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_ASIDE_HH
