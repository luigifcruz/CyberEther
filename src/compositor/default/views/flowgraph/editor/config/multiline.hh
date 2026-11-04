#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH

#include "types.hh"

#include "../field_tooltip.hh"

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
        parsedEncoded = std::move(nextValue);
        const std::string& shown = this->config.draft.value_or(parsedEncoded);
        help.update(this->config.id, this->config.help);
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .tooltip = help.overlay(),
            .content = &editor,
        });
        const bool dirty = shown != parsedEncoded;
        editor.update({
            .id = this->config.id + "Editor",
            .value = shown,
            .status = dirty ? "Edited. Press Ctrl+Enter to apply." : this->config.status,
            .height = 200.0f,
            .statusTone = dirty ? Sakura::StatusTone::Info : this->config.statusTone,
            .collapsible = collapsible,
            .backgroundColorKey = "card",
            .onChange = [this](std::string nextValue) {
                if (this->config.onDraft) {
                    this->config.onDraft(std::move(nextValue), parsedEncoded);
                }
            },
            .onSubmit = [this](std::string nextValue) {
                if (this->config.onDraft) {
                    this->config.onDraft(nextValue, parsedEncoded);
                }
                Parser::Map patch;
                patch[this->config.name] = std::move(nextValue);
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false, {});
                }
            },
        });
        return Result::SUCCESS;
    }

    Sakura::Component& component() {
        return frame;
    }

 private:
    Config config;
    Parser::Map parsedFormat;
    std::string parsedEncoded;
    bool collapsible = false;
    FlowgraphFieldTooltip help;
    Sakura::NodeField frame;
    Sakura::CodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MULTILINE_HH
