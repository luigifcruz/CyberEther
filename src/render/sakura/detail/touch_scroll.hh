#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_TOUCH_SCROLL_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_TOUCH_SCROLL_HH

#include <jetstream/surface.hh>

#include <memory>
#include <optional>
#include <vector>

namespace Jetstream::Sakura {

class Component;

class TouchScroll {
 public:
    using Handler = bool (*)(Component&, const MouseEvent&);

    bool begin(Component& root, const MouseEvent& event);
    void send(const MouseEvent& event);
    void reset();
    bool active() const { return owner.has_value(); }

    static bool dispatch(Component& component, const MouseEvent& event, Handler handler);

 private:
    struct Link {
        Component* component;
        std::weak_ptr<bool> alive;
        U64 attachment;
        U64 hidden;
    };
    struct Owner {
        Component* component;
        Handler handler;
        Extent2D<F32> position;
        std::vector<Link> path;
        bool ended = false;
    };
    struct Capture;
    struct Delivery;
    static thread_local Capture* capture;
    static thread_local Delivery* delivery;
    std::optional<Owner> owner;
    Component* root = nullptr;
    std::weak_ptr<bool> rootAlive;
    U64 generation = 0;

    bool valid() const;
    static void cancel(const Owner& owner);
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_TOUCH_SCROLL_HH
