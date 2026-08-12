#include <jetstream/render/sakura/components/combo.hh>

#include "../helpers.hh"

namespace Jetstream::Sakura {

struct Combo::Impl {
    Config config;
};

Combo::Combo() {
    this->impl = std::make_unique<Impl>();
}

Combo::~Combo() = default;
Combo::Combo(Combo&&) noexcept = default;
Combo& Combo::operator=(Combo&&) noexcept = default;

bool Combo::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Combo::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    const std::string comboId = "##" + config.id;
    ImGui::SetNextItemWidth(config.width > 0.0f ? Scale(ctx, config.width) : -FLT_MIN);

    const char* preview = config.value.empty() ? "Select" : config.value.c_str();
    if (config.disabled) {
        ImGui::BeginDisabled();
    }

    if (ImGui::BeginCombo(comboId.c_str(), preview)) {
        const ImGuiStyle& style = ImGui::GetStyle();
        const float popupInset = style.WindowPadding.x > style.FramePadding.x
                                     ? style.WindowPadding.x - style.FramePadding.x
                                     : 0.0f;
        if (popupInset > 0.0f) {
            ImGui::Indent(popupInset);
        }
        const float availableWidth = ImGui::GetContentRegionAvail().x;
        const float itemWidth = availableWidth > popupInset ? availableWidth - popupInset : 0.0f;
        for (const auto& option : config.options) {
            const bool selected = config.value == option;
            const bool activated = popupInset > 0.0f
                                       ? ImGui::Selectable(option.c_str(), selected, ImGuiSelectableFlags_None, ImVec2(itemWidth, 0.0f))
                                       : ImGui::Selectable(option.c_str(), selected);
            if (activated) {
                if (config.onChange) {
                    config.onChange(option);
                }
            }

            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        if (popupInset > 0.0f) {
            ImGui::Unindent(popupInset);
        }
        ImGui::EndCombo();
    }

    if (config.disabled) {
        ImGui::EndDisabled();
    }
}

}  // namespace Jetstream::Sakura
