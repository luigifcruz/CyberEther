#ifndef JETSTREAM_RENDER_SAKURA_NODE_DISPLAY_HH
#define JETSTREAM_RENDER_SAKURA_NODE_DISPLAY_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/surface.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Render {
class Texture;
}  // namespace Jetstream::Render

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

struct NodeProgressBar : public Component {
    struct Config {
        std::string id;
        F32 value = 0.0f;
        std::string overlay;
        std::string colorKey = "button_active";

        bool operator==(const Config&) const = default;
    };

    NodeProgressBar();
    ~NodeProgressBar();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct NodeTable : public Component {
    struct Config {
        std::string id;
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
        bool showHeaders = true;

        bool operator==(const Config&) const = default;
    };

    NodeTable();
    ~NodeTable();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct NodeSurface : public Component {
    struct Config {
        std::string id;
        std::shared_ptr<const Render::Texture> texture;
        F32 minimumHeight = 150.0f;
        bool detachOverlay = false;
        Callback<void(const SurfaceResize&)> onSize;
        Callback<void()> onDetach;

        bool operator==(const Config&) const = default;
    };

    NodeSurface();
    ~NodeSurface();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_DISPLAY_HH
