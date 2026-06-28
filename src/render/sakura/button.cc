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

    if (config.variant == Variant::Text) {
        if (config.disabled) {
            ImGui::BeginDisabled();
        }

        if (config.textScale != 1.0f) {
            ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * config.textScale);
        }

        const ImVec4 textColor = Private::ImColor(ctx, config.textColorKey.empty()
                                                       ? (config.disabled ? "text_disabled" : "text_primary")
                                                       : config.textColorKey);
        const ImU32 textColorU32 = ImGui::GetColorU32(textColor);
        const ImVec2 cursor = ImGui::GetCursorScreenPos();
        const ImVec2 textSize = ImGui::CalcTextSize(config.str.c_str());
        ImVec2 buttonSize = Private::ToImVec2(Scale(ctx, config.size));
        if (buttonSize.x <= 0.0f) {
            buttonSize.x = textSize.x;
        }
        if (buttonSize.y <= 0.0f) {
            buttonSize.y = textSize.y;
        }

        const bool pressed = ImGui::InvisibleButton("##text-button", buttonSize);
        const bool hovered = !config.disabled && ImGui::IsItemHovered();
        if (hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddText(cursor, textColorU32, config.str.c_str());
        if (hovered) {
            const F32 underlineY = cursor.y + textSize.y;
            drawList->AddLine(ImVec2(cursor.x, underlineY),
                              ImVec2(cursor.x + textSize.x, underlineY),
                              textColorU32,
                              Scale(ctx, 1.0f));
        }

        if (pressed && config.onClick) {
            config.onClick();
        }

        if (config.textScale != 1.0f) {
            ImGui::PopFont();
        }

        if (config.disabled) {
            ImGui::EndDisabled();
        }

        ImGui::PopID();
        return;
    }

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
    const auto colorKey = config.colorKey.empty() ? defaultColorKey : config.colorKey;
    const auto hoveredColorKey = config.hoveredColorKey.empty() ? defaultHoveredColorKey : config.hoveredColorKey;
    const auto activeColorKey = config.activeColorKey.empty() ? defaultActiveColorKey : config.activeColorKey;
    const auto textColorKey = config.textColorKey.empty() ? defaultTextColorKey : config.textColorKey;
    const auto borderColorKey = config.borderColorKey.empty() ? defaultBorderColorKey : config.borderColorKey;

    if (config.variant != Variant::Default || customColors) {
        ImGui::PushStyleColor(ImGuiCol_Button, Private::ImColor(ctx, colorKey));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Private::ImColor(ctx, hoveredColorKey));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, Private::ImColor(ctx, activeColorKey));
        styleColorCount += 3;
    }

    ImGui::PushStyleColor(ImGuiCol_Text, Private::ImColor(ctx, textColorKey));
    styleColorCount += 1;

    ImGui::PushStyleColor(ImGuiCol_Border, Private::ImColor(ctx, borderColorKey));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, Scale(ctx, 1.0f));
    styleColorCount += 1;

    if (config.disabled) {
        ImGui::BeginDisabled();
    }

    if (config.textScale != 1.0f) {
        ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * config.textScale);
    }

    const bool pressed = ImGui::Button(config.str.c_str(), Private::ToImVec2(Scale(ctx, config.size)));
    if (!config.disabled && ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }
    if (pressed) {
        if (config.onClick) {
            config.onClick();
        }
    }

    if (config.textScale != 1.0f) {
        ImGui::PopFont();
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
