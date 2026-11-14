#include "modal_header.hh"

namespace Jetstream {

void ModalHeader::update(Config config) {
    using Text = Sakura::Text;

    title.update({
        .str = config.title,
        .font = Text::Font::Display,
        .scale = Sakura::Typography::Scale::Heading,
    });
    description.update({
        .str = config.description,
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
    });
    divider.update({.spacing = config.dividerSpacing});
    std::vector<Item> items = {{.child = &title}};
    if (!config.description.empty()) {
        items.push_back({.child = &description});
    }
    items.push_back({.child = &divider});
    Stack::update({.items = std::move(items)});
}

}  // namespace Jetstream
