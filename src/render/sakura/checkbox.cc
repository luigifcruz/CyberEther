#include <jetstream/render/sakura/checkbox.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Checkbox::Impl {
    Config config;
};

Checkbox::Checkbox() {
    this->impl = std::make_unique<Impl>();
}

Checkbox::~Checkbox() = default;
Checkbox::Checkbox(Checkbox&&) noexcept = default;
Checkbox& Checkbox::operator=(Checkbox&&) noexcept = default;

bool Checkbox::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Checkbox::render(const Context& ctx) const {
    (void)ctx;
    const auto& config = this->impl->config;

    const char* label = config.label.c_str();
    ImGui::PushID(config.id.c_str());

    bool value = config.value;
    const bool changed = ImGui::Checkbox(label, &value);
    if (changed && config.onChange) {
        config.onChange(value);
    }

    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
