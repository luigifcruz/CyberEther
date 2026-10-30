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
            .title = false,
            .background = false,
            .content = &editor,
        });
        const bool dirty = buffer != parsedEncoded;
        editor.update({
            .id = this->config.id + "Editor",
            .value = buffer,
            .consoleOutput = this->config.consoleOutput,
            .status = dirty ? "Edited. Press Ctrl+Enter to run." : this->config.status,
            .height = 0.0f,
            .statusTone = dirty ? Sakura::StatusTone::Info : this->config.statusTone,
            .consoleVisible = this->config.consoleVisible,
            .maxAutoHeightWindowRatio = 0.65f,
            .language = Sakura::CodeEditor::Language::Python,
            .lineNumbers = true,
            .lineWrapping = false,
            .fontSize = 15.0f,
            .onChange = [this](std::string nextValue) {
                buffer = std::move(nextValue);
            },
            .onSubmit = [this](std::string nextValue) {
                buffer = std::move(nextValue);
                Parser::Map patch;
                patch[this->config.name] = buffer;
                if (this->config.onApply) {
                    this->config.onApply(std::move(patch), false, {});
                }
            },
            .onContentOverflow = this->config.onContentOverflow,
        });
        return Result::SUCCESS;
    }

    Sakura::Component& component() {
        return frame;
    }

 private:
    Config config;
    std::string parsedEncoded;
    std::string buffer;
    Sakura::NodeField frame;
    Sakura::CodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_PYTHON_HH
