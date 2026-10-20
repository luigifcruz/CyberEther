#ifndef JETSTREAM_RENDER_SAKURA_NODE_FIELD_HH
#define JETSTREAM_RENDER_SAKURA_NODE_FIELD_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeOverlay {
    Callback<Component*()> content;
    U64 revision = 0;

    explicit operator bool() const {
        return static_cast<bool>(content);
    }

    Component* operator()() const {
        return content();
    }

    bool operator==(const NodeOverlay&) const = default;
};

struct NodeField : public Component {
    static constexpr F32 Gap = 2.0f;

    struct Config {
        std::string id;
        std::string label;
        NodeOverlay tooltip;
        bool title = true;
        bool background = true;
        Component* content = nullptr;

        bool operator==(const Config&) const = default;
    };

    NodeField();
    ~NodeField();

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

#endif  // JETSTREAM_RENDER_SAKURA_NODE_FIELD_HH
