#ifndef JETSTREAM_RENDER_SAKURA_SHELL_HH
#define JETSTREAM_RENDER_SAKURA_SHELL_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/root.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

enum class Anchor {
    Fill,
    TopLeft,
    TopCenter,
    TopRight,
    Center,
    BottomLeft,
    BottomCenter,
    BottomRight,
};

struct Overlay : public Component {
    struct Item {
        Component* child = nullptr;
        Anchor anchor = Anchor::Fill;
        Extent2D<F32> offset = {0.0f, 0.0f};
        Extent2D<F32> size = {0.0f, 0.0f};

        bool operator==(const Item&) const = default;
    };

    struct Config {
        std::vector<Item> items;

        bool operator==(const Config&) const = default;
    };

    Overlay();
    ~Overlay();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct HudLayer : public Layer {
    struct Hud {
        Component* content = nullptr;
        Anchor anchor = Anchor::TopLeft;
        F32 margin = 12.0f;
        Extent2D<F32> padding = {12.0f, 12.0f};
        F32 borderSize = 0.5f;
        F32 rounding = 12.0f;
        std::string backgroundColorKey = "background";
        std::string borderColorKey = "border";
        bool clickable = false;
        Callback<void()> onClick;
        Extent2D<F32> size = {0.0f, 0.0f};
        Extent2D<F32> offset = {0.0f, 0.0f};

        bool operator==(const Hud&) const = default;
    };

    struct Config {
        std::vector<Hud> huds;

        bool operator==(const Config&) const = default;
    };

    HudLayer();
    ~HudLayer();

    void update(Config config);
    bool covers(const Extent2D<F32>& point) const override;

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct ModalLayer : public Layer {
    struct Config {
        Component* content = nullptr;
        Extent2D<F32> size = {550.0f, 0.0f};
        Callback<void()> onClose;

        bool operator==(const Config&) const = default;
    };

    ModalLayer();
    ~ModalLayer();

    void update(Config config);
    bool covers(const Extent2D<F32>& point) const override;
    bool modal() const override;

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

    bool shortcutEvent(const KeyEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct FloatingLayer : public Layer {
    static constexpr F32 Rounding = 12.0f;

    struct Window {
        std::string id;
        std::string title;
        Component* content = nullptr;
        Extent2D<F32> size = {500.0f, 400.0f};
        F32 padding = 0.0f;
        Callback<void()> onClose;
        Callback<void()> onFocus;

        bool operator==(const Window&) const = default;
    };

    struct Config {
        std::vector<Window> windows;

        bool operator==(const Config&) const = default;
    };

    FloatingLayer();
    ~FloatingLayer();

    void update(Config config);
    bool covers(const Extent2D<F32>& point) const override;

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    bool deliverTo(const std::string& id, const MouseEvent& event);
    void pointTo(const std::string& id, const MouseEvent& event);
};

struct ToastLayer : public Layer {
    ToastLayer();
    ~ToastLayer();

    void update();
    bool covers(const Extent2D<F32>& point) const override;

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct BackgroundLayer : public Layer {
    struct Config {
        bool particles = true;
        U64 particleCount = 60;
        F32 topOffset = 0.0f;
        std::string backgroundColorKey = "background";
        std::string particleColorKey = "workspace_particle";

        bool operator==(const Config&) const = default;
    };

    BackgroundLayer();
    ~BackgroundLayer();

    void update(Config config);
    bool covers(const Extent2D<F32>& point) const override;

 protected:
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SHELL_HH
