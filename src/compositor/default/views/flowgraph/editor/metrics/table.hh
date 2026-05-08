#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH

#include "types.hh"

#include <sstream>

namespace Jetstream {

struct FlowgraphMetricTable : public Sakura::Component {
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
        errorText.clear();
        if (!config.value.has_value()) {
            errorText = "No metric";
            columns.clear();
            rows.clear();
            parsedValue.clear();
            return;
        }

        std::string value;
        try {
            value = std::any_cast<std::string>(config.value);
        } catch (const std::bad_any_cast&) {
            errorText = "Invalid metric type";
            columns.clear();
            rows.clear();
            parsedValue.clear();
            return;
        }

        if (value == parsedValue) {
            return;
        }

        parsedValue = value;
        columns.clear();
        rows.clear();

        std::istringstream stream(value);
        std::string line;
        while (std::getline(stream, line)) {
            if (line.empty()) continue;

            std::vector<std::string> cols;
            std::istringstream lineStream(line);
            std::string cell;
            while (std::getline(lineStream, cell, '\t')) {
                cols.push_back(cell);
            }

            if (cols.empty()) {
                continue;
            }
            if (columns.empty()) {
                columns = std::move(cols);
            } else {
                rows.push_back(std::move(cols));
            }
        }

        if (columns.empty()) {
            errorText = "No data.";
        }
    }

    Config config;
    std::string parsedValue;
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    std::string errorText;
    Sakura::NodeField frame;
    Sakura::NodeLabel error;
    Sakura::NodeTable table;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_METRICS_TABLE_HH
