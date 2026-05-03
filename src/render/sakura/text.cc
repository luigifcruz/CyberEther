#include <jetstream/render/sakura/text.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Text::Impl {
    Config config;

    std::string textColorKey() const {
        if (!config.colorKey.empty()) {
            return config.colorKey;
        }

        switch (config.tone) {
            case Tone::Primary: return "text_primary";
            case Tone::Secondary: return "text_secondary";
            case Tone::Disabled: return "text_disabled";
            case Tone::Accent: return "accent_color";
            case Tone::Success: return "success_green";
            case Tone::Warning: return "warning_yellow";
        }
        return "text_primary";
    }

    ImFont* resolveFont(const Context& ctx) const {
        switch (config.font) {
            case Font::Body: return Private::NativeFont(ctx.fonts.body) ? Private::NativeFont(ctx.fonts.body) : ImGui::GetFont();
            case Font::H1: return Private::NativeFont(ctx.fonts.h1) ? Private::NativeFont(ctx.fonts.h1) : ImGui::GetFont();
            case Font::H2: return Private::NativeFont(ctx.fonts.h2) ? Private::NativeFont(ctx.fonts.h2) : ImGui::GetFont();
            case Font::Bold: return Private::NativeFont(ctx.fonts.bold) ? Private::NativeFont(ctx.fonts.bold) : ImGui::GetFont();
            case Font::Current: return nullptr;
        }
        return nullptr;
    }
};

Text::Text() {
    this->impl = std::make_unique<Impl>();
}

Text::~Text() = default;
Text::Text(Text&&) noexcept = default;
Text& Text::operator=(Text&&) noexcept = default;

bool Text::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Text::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleColor(ImGuiCol_Text, Private::ImColor(ctx, this->impl->textColorKey(), ImGui::GetStyleColorVec4(ImGuiCol_Text)));

    if (config.font != Font::Current || config.scale != 1.0f) {
        ImGui::PushFont(this->impl->resolveFont(ctx), ImGui::GetStyle().FontSizeBase * config.scale);
    }

    if (config.align != Align::Left && !config.wrapped) {
        const F32 startX = ImGui::GetCursorPosX();
        const F32 availableWidth = ImGui::GetContentRegionAvail().x;
        const char* lineStart = config.str.c_str();
        const char* textEnd = lineStart + config.str.size();

        while (lineStart <= textEnd) {
            const char* lineEnd = std::find(lineStart, textEnd, '\n');
            const F32 textWidth = ImGui::CalcTextSize(lineStart, lineEnd).x;
            const F32 offset = config.align == Align::Center ?
                               (availableWidth - textWidth) * 0.5f :
                               availableWidth - textWidth;
            ImGui::SetCursorPosX(startX + std::max(0.0f, offset));
            ImGui::TextUnformatted(lineStart, lineEnd);

            if (lineEnd == textEnd) {
                break;
            }
            lineStart = lineEnd + 1;
        }
    } else if (config.wrapped) {
        ImGui::TextWrapped("%s", config.str.c_str());
    } else {
        ImGui::TextUnformatted(config.str.c_str());
    }

    if (config.font != Font::Current || config.scale != 1.0f) {
        ImGui::PopFont();
    }

    ImGui::PopStyleColor();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
