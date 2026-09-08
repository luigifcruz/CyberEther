#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigMultilineField {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config) {
        this->config = std::move(config);
        if (this->config.format != parsedFormat) {
            collapsible = Parser::Get<bool>(this->config.format, "collapsible", false);
            parsedFormat = this->config.format;
        }
        std::string nextValue;
        JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, nextValue));
        if (nextValue != parsedEncoded) {
            buffer = nextValue;
            parsedEncoded = nextValue;
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
                Parser::Map patch;
                patch[this->config.name] = buffer;
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false);
                }
            },
        });
        return Result::SUCCESS;
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            editor.render(ctx);
        });
    }

 private:
    Config config;
    Parser::Map parsedFormat;
    std::string parsedEncoded;
    std::string buffer;
    bool collapsible = false;
    Sakura::NodeField frame;
    Sakura::NodeCodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH
