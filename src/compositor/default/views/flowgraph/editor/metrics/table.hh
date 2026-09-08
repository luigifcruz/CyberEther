#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphMetricTable {
    using Config = FlowgraphMetricConfig;

    void update(Config config) {
        this->config = std::move(config);
        parseValue();
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
            .background = false,
        });
        table.update({
            .id = this->config.id + "Table",
            .columns = columns,
            .rows = rows,
            .showHeaders = true,
        });
        error.update({
            .id = this->config.id + "Error",
            .str = errorText,
            .tone = Sakura::Text::Tone::Warning,
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            if (!errorText.empty()) {
                error.render(ctx);
            } else {
                table.render(ctx);
            }
        });
    }

 private:
    void parseValue() {
        if (!config.value.has_value()) {
            errorText = "No metric";
            columns.clear();
            rows.clear();
            parsedValue.reset();
            return;
        }

        const auto* value = std::any_cast<Parser::Map>(&config.value);
        if (!value) {
            errorText = "Invalid metric type";
            columns.clear();
            rows.clear();
            parsedValue.reset();
            return;
        }

        if (parsedValue && *value == *parsedValue) {
            return;
        }

        parsedValue = *value;
        errorText.clear();
        columns.clear();
        rows.clear();
        if (Parser::Deserialize(*value, "columns", columns) != Result::SUCCESS ||
            Parser::Deserialize(*value, "rows", rows) != Result::SUCCESS) {
            errorText = "Invalid table data";
            return;
        }
        if (std::any_of(rows.begin(), rows.end(), [&](const auto& row) { return row.size() != columns.size(); })) {
            errorText = "Invalid table row size";
            return;
        }

        if (columns.empty()) {
            errorText = "No data.";
        }
    }

    Config config;
    std::optional<Parser::Map> parsedValue;
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    std::string errorText;
    Sakura::NodeField frame;
    Sakura::NodeLabel error;
    Sakura::NodeTable table;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH
