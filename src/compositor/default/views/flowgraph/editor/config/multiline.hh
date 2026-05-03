#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigMultilineField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            const auto parts = Parser::SplitString(this->config.format, ":");
            collapsible = parts.size() > 1 && parts[1] == "collapsible";
            parsedFormat = this->config.format;
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        editor.update({
            .id = this->config.id + "Editor",
            .value = this->config.encoded,
            .collapsible = collapsible,
            .onChange = [this](std::string value) {
                auto values = this->config.values;
                values[this->config.name] = std::move(value);
                if (this->config.onApply) {
                    this->config.onApply(std::move(values), false);
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            editor.render(ctx);
        });
    }

 private:
    Config config;
    std::string parsedFormat;
    bool collapsible = false;
    Sakura::NodeField frame;
    Sakura::NodeCodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH
