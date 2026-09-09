#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH

#include "types.hh"

namespace Jetstream {

struct FlowgraphConfigPythonField {
    using Config = FlowgraphConfigFieldConfig;

    FlowgraphNodeHeightSpec heightSpec() const {
        return {
            .policy = FlowgraphNodeHeightPolicy::FillRemaining,
            .minimum = 120.0f,
        };
    }

    Result update(Config config) {
        this->config = std::move(config);
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
            .title = false,
            .background = false,
        });
        updateEditor();
        return Result::SUCCESS;
    }

    void setAllocatedHeight(std::optional<F32> height) {
        allocatedHeight = height;
        updateEditor();
    }

    void render(const Sakura::Context& ctx) const {
        frame.render(ctx, [this](const Sakura::Context& ctx) {
            editor.render(ctx);
        });
    }

 private:
    void updateEditor() {
        const bool dirty = buffer != parsedEncoded;
        editor.update({
            .id = this->config.id + "Editor",
            .value = buffer,
            .consoleOutput = this->config.consoleOutput,
            .status = dirty ? "Edited. Press Ctrl+Enter to run." : this->config.status,
            .statusTone = dirty ? Sakura::NodeCodeEditor::StatusTone::Info : this->config.statusTone,
            .consoleVisible = this->config.consoleVisible,
            .autoHeight = true,
            .height = allocatedHeight,
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

    Config config;
    std::optional<F32> allocatedHeight;
    std::string parsedEncoded;
    std::string buffer;
    Sakura::NodeField frame;
    Sakura::NodeCodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH
