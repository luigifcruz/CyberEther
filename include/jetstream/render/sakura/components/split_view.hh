#ifndef JETSTREAM_RENDER_SAKURA_SPLIT_VIEW_HH
#define JETSTREAM_RENDER_SAKURA_SPLIT_VIEW_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>

namespace Jetstream::Sakura {

struct SplitView : public Component {
    struct Config {
        Component* left = nullptr;
        Component* right = nullptr;
        F32 leftWidth = 220.0f;
        F32 height = 0.0f;

        bool operator==(const Config&) const = default;
    };

    SplitView();
    ~SplitView();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SPLIT_VIEW_HH
