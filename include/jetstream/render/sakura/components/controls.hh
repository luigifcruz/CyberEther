#ifndef JETSTREAM_RENDER_SAKURA_CONTROLS_HH
#define JETSTREAM_RENDER_SAKURA_CONTROLS_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Button : public Component {
    enum class Variant {
        Default,
        Action,
        Destructive,
        Text,
    };

    struct Config {
        std::string str;
        Extent2D<F32> size = {0.0f, 0.0f};
        Variant variant = Variant::Default;
        bool disabled = false;
        std::string colorKey;
        std::string hoveredColorKey;
        std::string activeColorKey;
        std::string borderColorKey;
        std::string textColorKey;
        F32 textScale = 1.0f;
        F32 rounding = 8.0f;
        Extent2D<F32> padding = {12.0f, 5.0f};
        F32 textAlign = 0.5f;
        Callback<void()> onClick;
        Callback<void()> onDoubleClick;

        bool operator==(const Config&) const = default;
    };

    Button();
    ~Button();

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

struct NavigationItem : public Button {
    struct Config {
        std::string label;
        bool selected = false;
        Callback<void()> onSelect;
        Callback<void()> onDoubleClick;

        bool operator==(const Config&) const = default;
    };

    void update(Config config);
};

struct TextInput : public Component {
    enum class Submit {
        OnEdit,
        OnEnter,
        OnCommit,
    };

    struct Config {
        std::string value;
        std::string hint;
        Submit submit = Submit::OnEnter;
        bool focus = false;
        bool selectAllOnFocus = false;
        bool multiline = false;
        F32 height = 0.0f;
        Callback<void(const std::string&)> onChange;
        Callback<void(const std::string&)> onSubmit;
        std::string backgroundColorKey = "card";
        Callback<void()> onCancel;
        Extent2D<F32> padding = {12.0f, 5.0f};

        bool operator==(const Config&) const = default;
    };

    TextInput();
    ~TextInput();

    void update(Config config);
    bool focused() const;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct Checkbox : public Component {
    struct Config {
        std::string label;
        bool value = false;
        Callback<void(bool)> onChange;

        bool operator==(const Config&) const = default;
    };

    Checkbox();
    ~Checkbox();

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

struct ProgressBar : public Component {
    struct Config {
        F32 value = 0.0f;
        std::string overlay;
        Extent2D<F32> size = {Fill, 20.0f};
        std::string colorKey = "action_btn";
        std::string backgroundColorKey;

        bool operator==(const Config&) const = default;
    };

    ProgressBar();
    ~ProgressBar();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct Combo : public Component {
    struct Config {
        std::string id;
        std::vector<std::string> options;
        std::string value;
        F32 width = Fill;
        bool disabled = false;
        std::optional<U64> selectedIndex;
        Callback<void(const std::string&)> onChange;
        Callback<void(U64)> onSelect;

        bool operator==(const Config&) const = default;
    };

    Combo();
    ~Combo();

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

struct SettingField : public Component {
    struct Config {
        std::string label;
        std::string description;
        Component* control = nullptr;
        bool divider = true;

        bool operator==(const Config&) const = default;
    };

    SettingField();
    ~SettingField();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct QrCode : public Component {
    struct Config {
        std::vector<U8> data;
        int width = 0;
        F32 moduleSize = 4.0f;
        Callback<void()> onClick;

        bool operator==(const Config&) const = default;
    };

    QrCode();
    ~QrCode();

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

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CONTROLS_HH
