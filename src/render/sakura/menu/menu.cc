#include <jetstream/render/sakura/menu/menu.hh>

#include "../base.hh"

namespace Jetstream::Sakura {

struct Menu::Impl {
    Config config;

    ImFont* resolveFont(const Context& ctx) const {
        switch (config.font) {
            case Text::Font::Body: return Private::NativeFont(ctx.fonts.body) ? Private::NativeFont(ctx.fonts.body) : ImGui::GetFont();
            case Text::Font::H1: return Private::NativeFont(ctx.fonts.h1) ? Private::NativeFont(ctx.fonts.h1) : ImGui::GetFont();
            case Text::Font::H2: return Private::NativeFont(ctx.fonts.h2) ? Private::NativeFont(ctx.fonts.h2) : ImGui::GetFont();
            case Text::Font::Bold: return Private::NativeFont(ctx.fonts.bold) ? Private::NativeFont(ctx.fonts.bold) : ImGui::GetFont();
            case Text::Font::Current: return ImGui::GetFont();
        }
        return ImGui::GetFont();
    }
};

Menu::Menu() {
    this->impl = std::make_unique<Impl>();
}

Menu::~Menu() = default;
Menu::Menu(Menu&&) noexcept = default;
Menu& Menu::operator=(Menu&&) noexcept = default;

bool Menu::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Menu::render(const Context& ctx, Child child) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    const F32 previousTextBaseOffset = window->DC.CurrLineTextBaseOffset;

    I32 colorCount = 0;
    ImGui::PushStyleColor(ImGuiCol_Text, Private::ImColor(ctx, config.colorKey));
    ++colorCount;

    I32 fontCount = 0;
    if (config.font != Text::Font::Current || config.scale != 1.0f) {
        ImGui::PushFont(this->impl->resolveFont(ctx), ImGui::GetStyle().FontSizeBase * config.scale);
        if (config.scale != 1.0f && window->DC.CurrLineSize.y > ImGui::GetFontSize()) {
            window->DC.CurrLineTextBaseOffset = std::max(0.0f,
                                                        (window->DC.CurrLineSize.y - ImGui::GetFontSize()) * 0.5f);
        }
        ++fontCount;
    }

    const bool opened = ImGui::BeginMenu(config.label.c_str(), config.enabled);
    window->DC.CurrLineTextBaseOffset = previousTextBaseOffset;

    if (fontCount > 0) {
        ImGui::PopFont();
    }
    if (colorCount > 0) {
        ImGui::PopStyleColor(colorCount);
    }

    if (opened) {
        if (child) {
            child(ctx);
        }
        ImGui::EndMenu();
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
