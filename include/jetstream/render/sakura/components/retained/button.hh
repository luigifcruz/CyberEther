#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_BUTTON_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_BUTTON_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura::Retained {

struct Button : public Component {
    struct Config {
        std::string id;
        std::string str;
        bool disabled = false;
        std::string colorKey = "button";
        std::string hoveredColorKey = "button_hovered";
        std::string activeColorKey = "button_active";
        std::string borderColorKey = "button_outline";
        std::string textColorKey = "button_text";
        F32 disabledAlpha = 0.4f;
        F32 fontSize = 15.0f;
        std::string fontName = "default_mono";
        F32 cornerRadius = 0.0f;
        F32 borderWidth = 0.0f;
        U64 maxCharacters = 64;
        std::function<void()> onClick;
    };

    Button();
    ~Button();

    bool update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool event(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_BUTTON_HH
