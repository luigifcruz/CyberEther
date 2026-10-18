#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_MODAL_HEADER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_MODAL_HEADER_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct ModalHeader : public Sakura::Stack {
    struct Config {
        std::string title;
        std::string description;
        F32 dividerSpacing = 4.0f;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        title.update({.str = config.title});
        description.update({
            .str = config.description,
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

 private:
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_MODAL_HEADER_HH
