#include <jetstream/render/sakura/components/hud.hh>

#include "../helpers.hh"

namespace Jetstream::Sakura {

struct Hud::Impl {
    Config config;
};

Hud::Hud() {
    this->impl = std::make_unique<Impl>();
}

Hud::~Hud() = default;
Hud::Hud(Hud&&) noexcept = default;
Hud& Hud::operator=(Hud&&) noexcept = default;

bool Hud::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Hud::render(const Context& ctx, Child child) const {
    const auto& config = this->impl->config;

    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const F32 padding = Scale(ctx, config.padding);
    ImVec2 position = Private::AnchorPoint(config.anchor, viewport->Pos, viewport->Size,
                                           {padding, padding, padding, padding});

    ImGui::SetNextWindowPos(position, ImGuiCond_Always, Private::AnchorPivot(config.anchor));
    ImGui::SetNextWindowViewport(viewport->ID);
    if (config.size.has_value()) {
        ImGui::SetNextWindowSize(Private::ToImVec2(Scale(ctx, *config.size)), ImGuiCond_Always);
    }

    I32 styleColorCount = 0;
    I32 styleVarCount = 0;
    ImVec4 backgroundColor = Private::ImColor(ctx, config.backgroundColorKey);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, backgroundColor);
    ImGui::PushStyleColor(ImGuiCol_PopupBg, backgroundColor);
    styleColorCount += 2;

    ImVec4 borderColor = Private::ImColor(ctx, config.borderColorKey);
    ImGui::PushStyleColor(ImGuiCol_Border, borderColor);
    styleColorCount += 1;
    if (config.windowPadding.has_value()) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, Private::ToImVec2(Scale(ctx, *config.windowPadding)));
        styleVarCount += 1;
    }
    if (config.borderSize.has_value()) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, Scale(ctx, *config.borderSize));
        styleVarCount += 1;
    }
    if (config.rounding.has_value()) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, Scale(ctx, *config.rounding));
        styleVarCount += 1;
    }

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration |
                             ImGuiWindowFlags_NoDocking |
                             ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav |
                             ImGuiWindowFlags_NoMove;
    flags |= ImGuiWindowFlags_Tooltip;

    const bool visible = ImGui::Begin(config.id.c_str(), nullptr, flags);
    if (styleVarCount > 0) {
        ImGui::PopStyleVar(styleVarCount);
    }
    if (styleColorCount > 0) {
        ImGui::PopStyleColor(styleColorCount);
    }

    if (visible && child) {
        child(ctx);
        const bool hovered = ImGui::IsWindowHovered();
        const bool clicked = hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left);
        if (config.clickable && hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }
        if (config.clickable && clicked && config.onClick) {
            config.onClick();
        }
    }
    ImGui::End();
}

}  // namespace Jetstream::Sakura
