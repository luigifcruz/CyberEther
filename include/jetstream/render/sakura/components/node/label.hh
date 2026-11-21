#ifndef JETSTREAM_RENDER_SAKURA_NODE_LABEL_HH
#define JETSTREAM_RENDER_SAKURA_NODE_LABEL_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeLabel : public Component {
    enum class Align : U8 {
        Left,
        Center,
        Right,
    };

    struct Config {
        std::string id;
        std::string str;
        std::string colorKey = "text_primary";
        Align align = Align::Left;
        bool boxed = false;

        bool operator==(const Config&) const = default;
    };

    NodeLabel();
    ~NodeLabel();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_LABEL_HH
