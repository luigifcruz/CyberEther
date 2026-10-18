#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <vector>

namespace Jetstream {

struct AboutInfoTable : public Sakura::Stack {
    struct Config {
        std::string id;
        std::string title;
        std::vector<std::vector<std::string>> rows;
    };

    void update(Config config) {
        title.update({.str = config.title});
        spacing.update({});
        table.update({
            .columns = {
                "Property",
                "Value",
            },
            .rows = std::move(config.rows),
            .fixedColumnWidths = {
                160.0f,
                0.0f,
            },
            .showHeaders = false,
        });
        divider.update({});
        Stack::update({
            .items = {
                {.child = &title},
                {.child = &spacing},
                {.child = &table},
                {.child = &divider},
            },
        });
    }

 private:
    Sakura::Text title;
    Sakura::Spacer spacing;
    Sakura::Table table;
    Sakura::Divider divider;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH
