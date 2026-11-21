#ifndef JETSTREAM_RENDER_SAKURA_NODE_PATH_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_PATH_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/node/field.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodePathInput : public Component {
    struct Config {
        std::string id;
        std::string value;
        NodeOverlay tooltip;
        Callback<void(const std::string&)> onChange;
        Callback<void()> onBrowse;

        bool operator==(const Config&) const = default;
    };

    NodePathInput();
    ~NodePathInput();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_PATH_INPUT_HH
