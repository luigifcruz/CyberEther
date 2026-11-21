#ifndef JETSTREAM_RENDER_SAKURA_NODE_CONTENT_STACK_HH
#define JETSTREAM_RENDER_SAKURA_NODE_CONTENT_STACK_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeContentStack : public Component {
    static constexpr F32 Spacing = 8.0f;

    struct Flex {
        F32 minimum = 0.0f;
        F32 grow = 1.0f;
        std::optional<F32> basis;

        bool operator==(const Flex&) const = default;
    };

    struct Item {
        Component* child = nullptr;
        std::optional<Flex> flex;
        std::optional<F32> gap;
        F32 trailingGap = 0.0f;

        bool operator==(const Item&) const = default;
    };

    struct Config {
        std::string id;
        std::vector<Item> items;

        bool operator==(const Config&) const = default;
    };

    NodeContentStack();
    ~NodeContentStack();

    void update(Config config);

    struct Metrics {
        F32 natural = 0.0f;
        F32 fixed = 0.0f;
        F32 minimum = 0.0f;
        F32 flexibleBasis = 0.0f;
    };

    bool empty() const;
    F32 leadingGap() const;
    Metrics metrics(const Context& ctx, F32 width);
    bool hasFlexible() const;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_CONTENT_STACK_HH
