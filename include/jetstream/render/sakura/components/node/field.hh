#ifndef JETSTREAM_RENDER_SAKURA_NODE_FIELD_HH
#define JETSTREAM_RENDER_SAKURA_NODE_FIELD_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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

struct NodeFieldGrid : public Component {
    struct Item {
        Component* child = nullptr;
        bool fullWidth = false;

        bool operator==(const Item&) const = default;
    };

    struct Config {
        std::string id;
        std::vector<Item> items;

        bool operator==(const Config&) const = default;
    };

    NodeFieldGrid();
    ~NodeFieldGrid();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

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

#endif  // JETSTREAM_RENDER_SAKURA_NODE_FIELD_HH
