#include "about_info_table.hh"

namespace Jetstream {

void AboutInfoTable::update(Config config) {
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
    SectionGroup::update({
        .title = std::move(config.title),
        .rows = std::move(rows),
    });
}

}  // namespace Jetstream
