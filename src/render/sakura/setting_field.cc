#include <jetstream/render/sakura/setting_field.hh>

#include <jetstream/render/sakura/divider.hh>
#include <jetstream/render/sakura/text.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct SettingField::Impl {
    Config config;
    Text labelText;
    Text descriptionText;
    Divider divider;
};

SettingField::SettingField() {
    this->impl = std::make_unique<Impl>();
}

SettingField::~SettingField() = default;
SettingField::SettingField(SettingField&&) noexcept = default;
SettingField& SettingField::operator=(SettingField&&) noexcept = default;

bool SettingField::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.labelText.update({
        .id = impl.config.id + "Label",
        .str = impl.config.label,
    });
    impl.descriptionText.update({
        .id = impl.config.id + "Description",
        .str = impl.config.description,
        .tone = Text::Tone::Disabled,
        .wrapped = true,
    });
    impl.divider.update({
        .id = impl.config.id + "Divider",
    });
    return true;
}

void SettingField::render(const Context& ctx, Child child) const {
    const auto& impl = *this->impl;
    const auto& config = impl.config;

    ImGui::PushID(config.id.c_str());

    if (!config.label.empty()) {
        impl.labelText.render(ctx);
    }

    if (child) {
        child(ctx);
    }

    if (!config.description.empty()) {
        impl.descriptionText.render(ctx);
    }

    if (config.divider) {
        impl.divider.render(ctx);
    }

    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
