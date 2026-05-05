#include <jetstream/render/sakura/menu/menu_bar.hh>

#include "../base.hh"

namespace Jetstream::Sakura {

struct MenuBar::Impl {
    Config config;
};

MenuBar::MenuBar() {
    this->impl = std::make_unique<Impl>();
}

MenuBar::~MenuBar() = default;
MenuBar::MenuBar(MenuBar&&) noexcept = default;
MenuBar& MenuBar::operator=(MenuBar&&) noexcept = default;

bool MenuBar::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void MenuBar::render(const Context& ctx, Child child) const {
    const auto& config = this->impl->config;

    const ImGuiStyle& style = ImGui::GetStyle();
    const F32 paddingY = style.FramePadding.y * config.heightScale;

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(style.FramePadding.x, paddingY));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    if (ImGui::BeginMainMenuBar()) {
        ImGui::PopStyleVar();
        if (child) {
            child(ctx);
        }
        if (config.onHeight) {
            config.onHeight(Unscale(ctx, ImGui::GetWindowSize().y));
        }
        ImGui::EndMainMenuBar();
    } else {
        ImGui::PopStyleVar();
    }
    ImGui::PopStyleVar();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
