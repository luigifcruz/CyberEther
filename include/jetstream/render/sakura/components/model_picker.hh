#ifndef JETSTREAM_RENDER_SAKURA_MODEL_PICKER_HH
#define JETSTREAM_RENDER_SAKURA_MODEL_PICKER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/style.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct ModelPicker : public Component {
    struct Option {
        std::string id;
        std::string label;
        std::string detail;

        bool operator==(const Option&) const = default;
    };

    struct Config {
        std::vector<Option> models;
        std::string value;
        std::string placeholder = "No models";
        std::vector<Option> efforts;
        std::string effort;
        std::string effortPlaceholder = "Select effort";
        bool disabled = false;
        F32 fontSize = Typography::FontSize;
        F32 rounding = 0.0f;
        F32 popupRounding = Rounding::Popup;
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
        Callback<void(const std::string&)> onSelect;
        Callback<void(const std::string&)> onEffort;

        bool operator==(const Config&) const = default;
    };

    ModelPicker();
    ~ModelPicker();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    void wentOffscreen(U64 frame) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MODEL_PICKER_HH
