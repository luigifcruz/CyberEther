#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigMarkdownField : public Sakura::Component {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.encoded != parsedEncoded) {
            value = this->config.encoded;
            parsedEncoded = this->config.encoded;
            if (!editing) {
                buffer = value;
            }
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
            .title = false,
            .background = false,
        });
        preview.update({
            .id = this->config.id + "Preview",
            .value = value,
        });
        editButton.update({
            .id = this->config.id + "Edit",
            .str = "Edit",
            .size = {-1.0f, 0.0f},
            .colorKey = "card",
            .hoveredColorKey = "card",
            .activeColorKey = "card",
            .onClick = [this]() {
                buffer = value;
                editing = true;
            },
        });
        doneButton.update({
            .id = this->config.id + "Done",
            .str = "Done",
            .size = {-1.0f, 0.0f},
            .colorKey = "card",
            .hoveredColorKey = "card",
            .activeColorKey = "card",
            .onClick = [this]() {
                auto values = this->config.values;
                values[this->config.name] = buffer;
                if (this->config.onApply) {
                    this->config.onApply(std::move(values), false);
                }
                editing = false;
            },
        });
        editor.update({
            .id = this->config.id + "Editor",
            .value = buffer,
            .onChange = [this](std::string nextValue) {
                buffer = std::move(nextValue);
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            if (!editing) {
                preview.render(ctx);
                editButton.render(ctx);
                return;
            }

            editor.render(ctx);
            doneButton.render(ctx);
        });
    }

 private:
    Config config;
    std::string parsedEncoded;
    std::string value;
    bool editing = false;
    std::string buffer;
    Sakura::NodeField frame;
    Sakura::NodeMarkdown preview;
    Sakura::NodeCodeEditor editor;
    Sakura::Button editButton;
    Sakura::Button doneButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_MARKDOWN_HH
