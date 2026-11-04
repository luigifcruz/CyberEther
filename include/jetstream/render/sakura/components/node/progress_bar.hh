#ifndef JETSTREAM_RENDER_SAKURA_NODE_PROGRESS_BAR_HH
#define JETSTREAM_RENDER_SAKURA_NODE_PROGRESS_BAR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeProgressBar : public Component {
    struct Config {
        std::string id;
        F32 value = 0.0f;
        std::string overlay;
        std::string colorKey = "button_active";

        bool operator==(const Config&) const = default;
    };

    NodeProgressBar();
    ~NodeProgressBar();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_PROGRESS_BAR_HH
