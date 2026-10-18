#ifndef JETSTREAM_RENDER_SAKURA_NODE_INPUTS_HH
#define JETSTREAM_RENDER_SAKURA_NODE_INPUTS_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/node/field.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeTextInput : public Component {
    enum class Submit : U8 {
        OnEnter,
        OnCommit,
    };

    struct Config {
        std::string id;
        std::string value;
        std::string hint;
        std::string unit;
        Submit submit = Submit::OnEnter;
        bool selectAllOnFocus = false;
        Callback<bool(U32)> characterFilter;
        bool background = true;
        std::string textColorKey = "text_primary";
        F32 height = 0.0f;
        Callback<void(const std::string&)> onChange;
        Callback<void()> onDeactivate;

        bool operator==(const Config&) const = default;
    };

    NodeTextInput();
    ~NodeTextInput();

    void update(Config config);
    bool focused() const;
    void focus(bool selectAll = false);
    void focusWhenPlaced(bool selectAll = false);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct NodeFloatInput : public Component {
    struct Config {
        std::string id;
        F32 value = 0.0f;
        std::string unit;
        I32 precision = 2;
        std::optional<F32> step;
        Callback<void(F32)> onChange;
        Callback<void(F32)> onStepChange;

        bool operator==(const Config&) const = default;
    };

    NodeFloatInput();
    ~NodeFloatInput();

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

struct NodeIntInput : public Component {
    struct Config {
        std::string id;
        I64 value = 0;
        std::string unit;
        Callback<void(I64)> onChange;

        bool operator==(const Config&) const = default;
    };

    NodeIntInput();
    ~NodeIntInput();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct NodeUIntInput : public Component {
    struct Config {
        std::string id;
        U64 value = 0;
        std::string unit;
        Callback<void(U64)> onChange;

        bool operator==(const Config&) const = default;
    };

    NodeUIntInput();
    ~NodeUIntInput();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct NodeBoolInput : public Component {
    struct Config {
        std::string id;
        bool value = false;
        Callback<void(bool)> onChange;

        bool operator==(const Config&) const = default;
    };

    NodeBoolInput();
    ~NodeBoolInput();

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

struct NodeRangeInput : public Component {
    struct Config {
        std::string id;
        F32 min = 0.0f;
        F32 max = 1.0f;
        F32 value = 0.0f;
        bool integer = false;
        std::string unit;
        I32 precision = 0;
        Callback<void(F32)> onChange;

        bool operator==(const Config&) const = default;
    };

    NodeRangeInput();
    ~NodeRangeInput();

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

struct NodeCombo : public Component {
    struct Config {
        std::string id;
        std::vector<std::string> options;
        std::string value;
        std::optional<U64> selectedIndex;
        Callback<void(const std::string&)> onChange;
        Callback<void(U64)> onSelect;

        bool operator==(const Config&) const = default;
    };

    NodeCombo();
    ~NodeCombo();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

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

#endif  // JETSTREAM_RENDER_SAKURA_NODE_INPUTS_HH
