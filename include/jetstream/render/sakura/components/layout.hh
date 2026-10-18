#ifndef JETSTREAM_RENDER_SAKURA_LAYOUT_HH
#define JETSTREAM_RENDER_SAKURA_LAYOUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Panel : public Component {
    struct Config {
        std::string id;
        Component* content = nullptr;
        Extent2D<F32> size = {Fill, 0.0f};
        F32 maxWidth = 0.0f;
        bool transparent = false;
        std::string backgroundColorKey = "background";

        bool operator==(const Config&) const = default;
    };

    Panel();
    ~Panel();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct Stack : public Component {
    static constexpr F32 Spacing = 8.0f;

    enum class Axis {
        Vertical,
        Horizontal,
    };

    struct Item {
        Component* child = nullptr;
        F32 size = 0.0f;
        F32 grow = 0.0f;

        bool operator==(const Item&) const = default;
    };

    struct Config {
        Axis axis = Axis::Vertical;
        F32 spacing = Spacing;
        bool centerCross = true;
        std::vector<Item> items;

        bool operator==(const Config&) const = default;
    };

    Stack();
    ~Stack();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct Spacer : public Component {
    struct Config {
        U64 lines = 1;

        bool operator==(const Config&) const = default;
    };

    Spacer();
    ~Spacer();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;

 private:
    Config config;
};

struct Divider : public Component {
    struct Config {
        F32 spacing = Stack::Spacing;
        bool separator = true;

        bool operator==(const Config&) const = default;
    };

    Divider();
    ~Divider();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct Div : public Component {
    struct Config {
        Component* content = nullptr;
        Extent2D<F32> size = {Fill, 0.0f};
        F32 padding = 0.0f;
        F32 rounding = 0.0f;
        bool border = true;
        F32 borderWidth = 1.0f;
        bool selected = false;
        std::string colorKey = "card";
        std::string hoveredColorKey = "header_hovered";
        std::string selectedColorKey = "header_hovered";
        std::string borderColorKey = "border";
        Callback<void()> onClick;
        Callback<void()> onDoubleClick;

        bool operator==(const Config&) const = default;
    };

    Div();
    ~Div();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct ScrollArea : public Component {
    struct Config {
        Component* content = nullptr;
        F32 height = 240.0f;
        bool alwaysScrollbar = true;
        bool border = true;

        bool operator==(const Config&) const = default;
    };

    ScrollArea();
    ~ScrollArea();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct TabBar : public Component {
    struct Config {
        std::vector<std::string> labels;
        std::vector<Component*> pages;

        bool operator==(const Config&) const = default;
    };

    TabBar();
    ~TabBar();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

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

struct Grid : public Component {
    struct Config {
        U64 columns = 2;
        Extent2D<F32> size = {Fill, 0.0f};
        Extent2D<F32> cellPadding = {4.0f, 2.0f};
        bool outerPadding = true;
        std::vector<Component*> items;

        bool operator==(const Config&) const = default;
    };

    Grid();
    ~Grid();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_LAYOUT_HH
