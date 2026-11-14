#ifndef JETSTREAM_RENDER_SAKURA_NODE_INPUT_ROW_HH
#define JETSTREAM_RENDER_SAKURA_NODE_INPUT_ROW_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeInputRow : public Component {
    struct Combo {
        std::vector<std::string> options;
        std::string value;
        F32 width = 0.0f;
        Callback<void(const std::string&)> onChange;

        bool operator==(const Combo&) const = default;
    };

    struct Config {
        std::string id;
        std::string shape;
        std::string shapeHint;
        F32 shapeMinWidth = 56.0f;
        Callback<void(const std::string&)> onShapeChange;
        std::string axes;
        std::string axesHint;
        F32 axesMinWidth = 88.0f;
        Callback<void(const std::string&)> onAxesChange;
        std::vector<Combo> combos;

        bool operator==(const Config&) const = default;
    };

    NodeInputRow();
    ~NodeInputRow();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_INPUT_ROW_HH
