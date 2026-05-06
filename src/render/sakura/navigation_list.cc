#include <jetstream/render/sakura/navigation_list.hh>

#include <jetstream/render/sakura/divider.hh>
#include <jetstream/render/sakura/navigation_item.hh>
#include <jetstream/render/sakura/spacing.hh>
#include <jetstream/render/sakura/text.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NavigationList::Impl {
    Config config;
    Text title;
    Divider divider;
    Spacing spacing;
    std::vector<NavigationItem> itemComponents;
};

NavigationList::NavigationList() {
    this->impl = std::make_unique<Impl>();
}

NavigationList::~NavigationList() = default;
NavigationList::NavigationList(NavigationList&&) noexcept = default;
NavigationList& NavigationList::operator=(NavigationList&&) noexcept = default;

bool NavigationList::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.title.update({
        .id = impl.config.id + "Title",
        .str = impl.config.title,
    });
    impl.divider.update({
        .id = impl.config.id + "Divider",
        .spacing = 0.0f,
    });
    impl.spacing.update({
        .id = impl.config.id + "Spacing",
    });

    impl.itemComponents.resize(impl.config.items.size());
    for (U64 i = 0; i < impl.itemComponents.size(); ++i) {
        impl.itemComponents[i].update({
            .id = impl.config.id + "Item" + std::to_string(i),
            .label = impl.config.items[i].label,
            .selected = impl.config.items[i].selected,
            .onSelect = [this, i]() {
                const auto& config = this->impl->config;
                if (i < config.items.size() && config.items[i].onSelect) {
                    config.items[i].onSelect();
                }
            },
        });
    }
    return true;
}

void NavigationList::render(const Context& ctx) const {
    const auto& impl = *this->impl;
    const auto& config = impl.config;

    ImGui::PushID(config.id.c_str());
    impl.title.render(ctx);
    impl.divider.render(ctx);
    impl.spacing.render(ctx);
    for (const auto& item : impl.itemComponents) {
        item.render(ctx);
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
