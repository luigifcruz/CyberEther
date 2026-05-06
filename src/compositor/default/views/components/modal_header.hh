#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_MODAL_HEADER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_MODAL_HEADER_HH

#include "jetstream/render/sakura/sakura.hh"

#include <string>

namespace Jetstream {

struct ModalHeader : public Sakura::Component {
    struct Config {
        std::string id;
        std::string title;
        std::string description;
    };

    void update(Config config) {
        this->config = std::move(config);
        title.update({
            .id = this->config.id + "Title",
            .str = this->config.title,
        });
        description.update({
            .id = this->config.id + "Description",
            .str = this->config.description,
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });
        divider.update({
            .id = this->config.id + "Divider",
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        if (!config.description.empty()) {
            description.render(ctx);
        }
        divider.render(ctx);
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_MODAL_HEADER_HH
