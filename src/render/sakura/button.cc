#include <jetstream/render/sakura/button.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Button::Impl {
    Config config;
};

Button::Button() {
    this->impl = std::make_unique<Impl>();
}

Button::~Button() = default;
Button::Button(Button&&) noexcept = default;
Button& Button::operator=(Button&&) noexcept = default;

bool Button::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Button::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());

    I32 styleColorCount = 0;
    const char* defaultColorKey = "button";
    const char* defaultHoveredColorKey = "button_hovered";
    const char* defaultActiveColorKey = "button_active";
    const char* defaultTextColorKey = "button_text";
    const char* defaultBorderColorKey = "button_outline";
    if (config.variant == Variant::Action) {
        defaultColorKey = "action_btn";
        defaultHoveredColorKey = "action_btn_hovered";
        defaultActiveColorKey = "action_btn_active";
        defaultTextColorKey = "action_btn_text";
        defaultBorderColorKey = "action_btn_outline";
    } else if (config.variant == Variant::Destructive) {
        defaultColorKey = "destructive_btn";
        defaultHoveredColorKey = "destructive_btn_hovered";
        defaultActiveColorKey = "destructive_btn_active";
        defaultTextColorKey = "destructive_btn_text";
        defaultBorderColorKey = "destructive_btn_outline";
    }

    const bool customColors = !config.colorKey.empty() ||
                              !config.hoveredColorKey.empty() ||
                              !config.activeColorKey.empty();
    if (config.variant != Variant::Default || customColors) {
        ImGui::PushStyleColor(ImGuiCol_Button, Private::ImColor(ctx, config.colorKey.empty() ? defaultColorKey : config.colorKey));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Private::ImColor(ctx, config.hoveredColorKey.empty() ? defaultHoveredColorKey : config.hoveredColorKey));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, Private::ImColor(ctx, config.activeColorKey.empty() ? defaultActiveColorKey : config.activeColorKey));
        styleColorCount += 3;
    }

    ImGui::PushStyleColor(ImGuiCol_Text, Private::ImColor(ctx, config.textColorKey.empty() ? defaultTextColorKey : config.textColorKey));
    styleColorCount += 1;

    ImGui::PushStyleColor(ImGuiCol_Border, Private::ImColor(ctx, config.borderColorKey.empty() ? defaultBorderColorKey : config.borderColorKey));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, Scale(ctx, 1.0f));
    styleColorCount += 1;

    if (config.disabled) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button(config.str.c_str(), Private::ToImVec2(Scale(ctx, config.size)))) {
        if (config.onClick) {
            config.onClick();
        }
    }

    if (config.disabled) {
        ImGui::EndDisabled();
    }

    ImGui::PopStyleVar();

    if (styleColorCount > 0) {
        ImGui::PopStyleColor(styleColorCount);
    }

    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
