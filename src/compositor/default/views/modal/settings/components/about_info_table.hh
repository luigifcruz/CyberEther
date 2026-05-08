#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH

#include "jetstream/render/sakura/sakura.hh"

#include <string>
#include <vector>

namespace Jetstream {

struct AboutInfoTable : public Sakura::Component {
    struct Config {
        std::string id;
        std::string title;
        std::vector<std::vector<std::string>> rows;
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = this->config.id + "Title",
            .str = this->config.title,
        });

        spacing.update({
            .id = this->config.id + "Spacing",
        });

        table.update({
            .id = this->config.id + "Table",
            .columns = {
                "Property",
                "Value",
            },
            .rows = this->config.rows,
            .fixedColumnWidths = {
                160.0f,
                0.0f,
            },
            .showHeaders = false,
        });

        divider.update({
            .id = this->config.id + "Divider",
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        spacing.render(ctx);
        table.render(ctx);
        divider.render(ctx);
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Spacing spacing;
    Sakura::Table table;
    Sakura::Divider divider;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH
