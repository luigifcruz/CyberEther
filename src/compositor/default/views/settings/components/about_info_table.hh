#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH

#include "settings_group.hh"

#include "jetstream/render/sakura/base.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct AboutInfoTable : public SettingsGroup {
    struct Config {
        std::string id;
        std::string title;
        std::vector<std::vector<std::string>> rows;
    };

    void update(Config config) {
        while (values.size() < config.rows.size()) {
            values.push_back(std::make_unique<Sakura::Text>());
        }
        std::vector<Sakura::SettingCard::Row> rows;
        rows.reserve(config.rows.size());
        for (U64 index = 0; index < config.rows.size(); ++index) {
            const auto& row = config.rows[index];
            const std::string label = row.empty() ? "" : row[0];
            const std::string value = row.size() > 1 ? row[1] : "";
            values[index]->update({
                .str = value,
                .font = Sakura::Text::Font::Sans,
                .align = Sakura::Text::Align::Right,
                .wrapped = true,
            });
            rows.push_back({.label = label, .control = values[index].get(), .wide = true});
        }
        SettingsGroup::update({
            .title = std::move(config.title),
            .rows = std::move(rows),
        });
    }

 private:
    std::vector<std::unique_ptr<Sakura::Text>> values;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_COMPONENTS_ABOUT_INFO_TABLE_HH
