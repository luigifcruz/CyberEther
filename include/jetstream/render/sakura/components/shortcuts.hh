#ifndef JETSTREAM_RENDER_SAKURA_SHORTCUTS_HH
#define JETSTREAM_RENDER_SAKURA_SHORTCUTS_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <vector>

namespace Jetstream::Sakura {

struct Shortcuts : public Component {
    enum class Modifier {
        Any,
        Primary,
        PrimaryAlt,
    };

    struct Binding {
        KeyCode key = KeyCode::Unknown;
        Modifier modifier = Modifier::Any;
        bool repeat = false;
        Callback<void()> onPressed;

        bool operator==(const Binding&) const = default;
    };

    struct Config {
        std::vector<Binding> bindings;
        bool enabled = true;
        Component* content = nullptr;

        bool operator==(const Config&) const = default;
    };

    Shortcuts();
    ~Shortcuts() override;

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool shortcutEvent(const KeyEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SHORTCUTS_HH
