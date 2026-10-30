#include <jetstream/render/sakura/components/setting_field.hh>

#include <jetstream/render/sakura/components/divider.hh>
#include <jetstream/render/sakura/components/stack.hh>
#include <jetstream/render/sakura/components/text.hh>

#include "../context.hh"

#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct SettingField::Impl {
    Config config;
    Text label;
    Text description;
    Divider divider;
    Stack stack;
};

SettingField::SettingField() {
    impl = std::make_unique<Impl>();
    addChild(impl->stack);
}

SettingField::~SettingField() = default;

void SettingField::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->label.update({.str = impl->config.label});
    impl->description.update({
        .str = impl->config.description,
        .tone = Text::Tone::Disabled,
        .wrapped = true,
    });
    impl->divider.update({});
    std::vector<Stack::Item> items;
    if (!impl->config.label.empty()) {
        items.push_back({.child = &impl->label});
    }
    if (impl->config.control) {
        items.push_back({.child = impl->config.control});
    }
    if (!impl->config.description.empty()) {
        items.push_back({.child = &impl->description});
    }
    if (impl->config.divider) {
        items.push_back({.child = &impl->divider});
    }
    impl->stack.update({.items = std::move(items)});
}

Extent2D<F32> SettingField::measure(const Context& ctx, Extent2D<F32> available) {
    return measureChild(impl->stack, ctx, available);
}

void SettingField::layout(const Context& ctx) {
    layoutChild(ctx, impl->stack, frame());
}

}  // namespace Jetstream::Sakura
