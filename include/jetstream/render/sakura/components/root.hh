#ifndef JETSTREAM_RENDER_SAKURA_ROOT_HH
#define JETSTREAM_RENDER_SAKURA_ROOT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

enum class PopupPlacement {
    Below,
    BelowEnd,
    Point,
    Side,
    Tooltip,
    Center,
};

struct Layer : public Component {
    struct Config {
        Component* content = nullptr;

        bool operator==(const Config&) const = default;
    };

    Layer();
    ~Layer();

    void update(Config config);
    virtual bool covers(const Extent2D<F32>& point) const;
    virtual bool modal() const;
    bool occludes(const Extent2D<F32>& point) const override;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    Component* content = nullptr;
};

struct Popup : public Component {
    struct Config {
        Component* content = nullptr;
        bool open = false;
        PopupPlacement placement = PopupPlacement::Below;
        bool atCursor = false;
        bool panel = false;
        F32 padding = 12.0f;
        F32 maxWidth = 0.0f;
        Extent2D<F32> size = {0.0f, 0.0f};
        Callback<void()> onDismiss;
        bool passthroughAnchor = false;

        bool operator==(const Config&) const = default;
    };

    Popup();
    ~Popup();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

struct Root {
    struct Config {
        std::string id;
        F32 topOffset = 0.0f;
        std::vector<Layer*> layers;

        bool operator==(const Config&) const = default;
    };

    Root();
    ~Root();

    Root(const Root&) = delete;
    Root& operator=(const Root&) = delete;

    void update(Config config);
    void render(const Sakura::Context& ctx);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_ROOT_HH
