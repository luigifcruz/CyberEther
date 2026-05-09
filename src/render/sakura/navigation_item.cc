#include <jetstream/render/sakura/navigation_item.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NavigationItem::Impl {
    Config config;
};

NavigationItem::NavigationItem() {
    this->impl = std::make_unique<Impl>();
}

NavigationItem::~NavigationItem() = default;
NavigationItem::NavigationItem(NavigationItem&&) noexcept = default;
NavigationItem& NavigationItem::operator=(NavigationItem&&) noexcept = default;

bool NavigationItem::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void NavigationItem::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleColor(ImGuiCol_Button, Private::ImColor(ctx, config.selected ? config.selectedColorKey : config.colorKey));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Private::ImColor(ctx, config.selected ? config.selectedColorKey : config.hoveredColorKey));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, Private::ImColor(ctx, config.activeColorKey));
    ImGui::PushStyleColor(ImGuiCol_Text, Private::ImColor(ctx, config.selected ? config.selectedTextColorKey : config.textColorKey));
    ImGui::PushStyleColor(ImGuiCol_Border, Private::ImColor(ctx, config.selected ? config.selectedBorderColorKey : config.borderColorKey));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, Private::ToImVec2(Scale(ctx, {12.0f, 10.0f})));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, Scale(ctx, 10.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, Scale(ctx, 1.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, Private::ToImVec2({0.0f, 0.5f}));

    if (ImGui::Button(config.label.c_str(), Private::ToImVec2(Scale(ctx, {-1.0f, 0.0f})))) {
        if (config.onSelect) {
            config.onSelect();
        }
    }

    ImGui::PopStyleVar(4);
    ImGui::PopStyleColor(5);
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
