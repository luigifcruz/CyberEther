#ifndef JETSTREAM_RENDER_SAKURA_NODE_FIELD_PANEL_HH
#define JETSTREAM_RENDER_SAKURA_NODE_FIELD_PANEL_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/node/content_stack.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeFieldPanel : public Component {
    struct Config {
        std::string id;
        NodeContentStack* content = nullptr;
        F32 height = 0.0f;

        bool operator==(const Config&) const = default;
    };

    NodeFieldPanel();
    ~NodeFieldPanel();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_FIELD_PANEL_HH
