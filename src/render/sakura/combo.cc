#include <jetstream/render/sakura/combo.hh>

#include "base.hh"

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
    (void)ctx;
    const auto& config = this->impl->config;

    const std::string comboId = "##" + config.id;
    ImGui::SetNextItemWidth(-FLT_MIN);

    const char* preview = config.value.empty() ? "Select" : config.value.c_str();
    if (ImGui::BeginCombo(comboId.c_str(), preview)) {
        for (const auto& option : config.options) {
            const bool selected = config.value == option;
            if (ImGui::Selectable(option.c_str(), selected)) {
                if (config.onChange) {
                    config.onChange(option);
                }
            }

            if (selected) {
                ImGui::SetItemDefaultFocus();
            }
        }
        ImGui::EndCombo();
    }
}

}  // namespace Jetstream::Sakura
