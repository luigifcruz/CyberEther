#ifndef JETSTREAM_RENDER_SAKURA_LAYER_HH
#define JETSTREAM_RENDER_SAKURA_LAYER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Layer {
    using Child = std::function<void(const Context&)>;

    using Anchor = Sakura::Anchor;

    struct Config {
        std::string id;
        Anchor anchor = Anchor::BottomRight;
        Extent2D<F32> size = {0.0f, 0.0f};
        F32 topOffset = 0.0f;
        U64 focusRequest = 0;
        U64 blurRequest = 0;
    };

    Layer();
    ~Layer();

    Layer(Layer&&) noexcept;
    Layer& operator=(Layer&&) noexcept;

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_LAYER_HH
