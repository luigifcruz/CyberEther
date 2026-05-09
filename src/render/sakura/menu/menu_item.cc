#include <jetstream/render/sakura/menu/menu_item.hh>

#include "../base.hh"

namespace Jetstream::Sakura {

struct MenuItem::Impl {
    Config config;
};

MenuItem::MenuItem() {
    this->impl = std::make_unique<Impl>();
}

MenuItem::~MenuItem() = default;
MenuItem::MenuItem(MenuItem&&) noexcept = default;
MenuItem& MenuItem::operator=(MenuItem&&) noexcept = default;

bool MenuItem::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void MenuItem::render(const Context& ctx) const {
    (void)ctx;
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    const char* shortcut = config.shortcut.empty() ? nullptr : config.shortcut.c_str();
    if (ImGui::MenuItem(config.label.c_str(), shortcut, config.selected, config.enabled)) {
        if (config.onClick) {
            config.onClick();
        }
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
