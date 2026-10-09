#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_DROPDOWN_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_DROPDOWN_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct Dropdown : public Component {
    struct Option {
        std::string id;
        std::string label;
        std::string detail;

        bool operator==(const Option&) const = default;
    };

    struct Config {
        std::string id;
        std::vector<Option> options;
        std::string value;
        std::string placeholder = "Select";
        bool disabled = false;
        bool popupAbove = false;
        bool popupAlignRight = false;
        bool fitValue = false;
        std::string colorKey = "button";
        std::string hoveredColorKey = "button_hovered";
        std::string activeColorKey = "button_active";
        std::string borderColorKey = "button_outline";
        std::string textColorKey = "button_text";
        std::string caretColorKey = "button_text";
        std::string selectedTextColorKey = "button_text";
        std::string detailColorKey = "text_secondary";
        std::string popupColorKey = "card";
        std::string popupBorderColorKey = "border";
        std::string rowHoveredColorKey = "button_hovered";
        F32 disabledAlpha = 0.4f;
        F32 fontSize = 15.0f;
        std::string fontName = "default_mono";
        F32 cornerRadius = 0.0f;
        F32 popupCornerRadius = 0.0f;
        F32 borderWidth = 0.0f;
        U64 maxCharacters = 64;
        std::function<void(const std::string&)> onSelect;
    };

    Dropdown();
    ~Dropdown();

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

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_DROPDOWN_HH
