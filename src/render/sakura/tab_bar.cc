#include <jetstream/render/sakura/tab_bar.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct TabBar::Impl {
    Config config;
};

TabBar::TabBar() {
    this->impl = std::make_unique<Impl>();
}

TabBar::~TabBar() = default;
TabBar::TabBar(TabBar&&) noexcept = default;
TabBar& TabBar::operator=(TabBar&&) noexcept = default;

bool TabBar::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void TabBar::render(const Context& ctx, Children children) const {
    (void)ctx;
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    if (ImGui::BeginTabBar(config.id.c_str(), ImGuiTabBarFlags_None)) {
        for (U64 i = 0; i < config.labels.size(); ++i) {
            if (ImGui::BeginTabItem(config.labels[i].c_str())) {
                if (i < children.size() && children[i]) {
                    children[i](ctx);
                }
                ImGui::EndTabItem();
            }
        }
        ImGui::EndTabBar();
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
