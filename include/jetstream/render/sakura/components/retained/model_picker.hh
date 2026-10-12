#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_MODEL_PICKER_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_MODEL_PICKER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/dropdown.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct ModelPicker : public Component {
    using Option = Dropdown::Option;

    struct Config {
        std::string id;
        std::vector<Option> models;
        std::string value;
        std::string placeholder = "No models";
        std::vector<Option> efforts;
        std::string effort;
        std::string effortPlaceholder = "Select effort";
        bool disabled = false;
        bool popupAbove = true;
        bool popupAlignRight = true;
        std::string colorKey = "transparent";
        std::string hoveredColorKey = "button_hovered";
        std::string activeColorKey = "button_active";
        std::string textColorKey = "text_primary";
        std::string detailColorKey = "text_secondary";
        std::string accentColorKey = "accent_color";
        std::string popupColorKey = "agent_composer";
        std::string popupBorderColorKey = "agent_outline";
        std::string rowHoveredColorKey = "button_hovered";
        std::string trackColorKey = "agent_outline";
        std::string thumbColorKey = "text_primary";
        F32 disabledAlpha = 0.4f;
        F32 fontSize = Typography::FontSize;
        std::string fontName = "default_body";
        F32 cornerRadius = 0.0f;
        F32 popupCornerRadius = 0.0f;
        U64 maxCharacters = 64;
        std::function<void(const std::string&)> onSelect;
        std::function<void(const std::string&)> onEffort;
    };

    ModelPicker();
    ~ModelPicker();

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

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_MODEL_PICKER_HH
