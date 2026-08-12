#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigPythonField {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.encoded != parsedEncoded) {
            buffer = this->config.encoded;
            parsedEncoded = this->config.encoded;
        }
        frame.update({
            .id = this->config.id,
            .label = this->config.label,
            .help = this->config.help,
            .title = false,
            .background = false,
        });
        const bool dirty = buffer != parsedEncoded;
        editor.update({
            .id = this->config.id + "Editor",
            .value = buffer,
            .consoleOutput = this->config.consoleOutput,
            .status = dirty ? "Edited. Press Ctrl+Enter to run." : this->config.status,
            .statusTone = dirty ? Sakura::NodeCodeEditor::StatusTone::Info : this->config.statusTone,
            .consoleVisible = this->config.consoleVisible,
            .autoHeight = true,
            .maxAutoHeightWindowRatio = 0.65f,
            .language = Sakura::NodeCodeEditor::Language::Python,
            .lineNumbers = true,
            .lineWrapping = false,
            .editorFontSize = 15.0f,
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
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            editor.render(ctx);
        });
    }

 private:
    Config config;
    std::string parsedEncoded;
    std::string buffer;
    Sakura::NodeField frame;
    Sakura::NodeCodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH
