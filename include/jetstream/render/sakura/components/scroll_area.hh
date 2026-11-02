#ifndef JETSTREAM_RENDER_SAKURA_SCROLL_AREA_HH
#define JETSTREAM_RENDER_SAKURA_SCROLL_AREA_HH

#include <jetstream/render/sakura/callback.hh>
#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct ScrollArea : public Component {
    struct Config {
        std::string id;
        Component* content = nullptr;
        F32 height = 240.0f;
        bool alwaysScrollbar = true;
        bool border = true;
        bool overlay = false;
        Extent2D<F32> scrollbarInset = {0.0f, 0.0f};
        Component* scrollTarget = nullptr;
        U64 scrollRequest = 0;
        Callback<void(F32)> onScroll;

        bool operator==(const Config&) const = default;
    };

    ScrollArea();
    ~ScrollArea();

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

#endif  // JETSTREAM_RENDER_SAKURA_SCROLL_AREA_HH
