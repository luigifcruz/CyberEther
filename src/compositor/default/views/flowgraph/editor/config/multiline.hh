#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigMultilineField {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            const auto parts = Parser::SplitString(this->config.format, ":");
            collapsible = parts.size() > 1 && parts[1] == "collapsible";
            parsedFormat = this->config.format;
        }
        if (this->config.encoded != parsedEncoded) {
            buffer = this->config.encoded;
            parsedEncoded = this->config.encoded;
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
        });
        const bool dirty = buffer != parsedEncoded;
        editor.update({
            .id = this->config.id + "Editor",
            .value = buffer,
            .status = dirty ? "Edited. Press Ctrl+Enter to apply." : this->config.status,
            .statusTone = dirty ? Sakura::NodeCodeEditor::StatusTone::Info : this->config.statusTone,
            .collapsible = collapsible,
            .backgroundColorKey = "card",
            .onChange = [this](std::string nextValue) {
                buffer = std::move(nextValue);
            },
            .onSubmit = [this](std::string nextValue) {
                buffer = std::move(nextValue);
                auto values = this->config.values;
                values[this->config.name] = buffer;
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
    std::string parsedEncoded;
    std::string buffer;
    bool collapsible = false;
    Sakura::NodeField frame;
    Sakura::NodeCodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH
