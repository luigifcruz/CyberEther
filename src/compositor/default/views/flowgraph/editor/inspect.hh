#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_INSPECT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_INSPECT_HH

#include "jetstream/parser.hh"
#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphNodeInspector {
    struct Config {
        std::string id;
        std::string name;
        Parser::Map value;
        std::function<void(Parser::Map)> onApply;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);

        editor.update({
            .id = this->config.id + ":yaml",
            .value = buffer,
            .status = status,
            .height = Sakura::Fill,
            .statusTone = statusTone,
            .backgroundColorKey = "tile_bg",
            .rounding = Sakura::FloatingLayer::Rounding,
            .onChange = [this](std::string value) {
                buffer = std::move(value);
                status = canApply()
                    ? "Edited. Press Cmd+Enter to apply."
                    : "Block configuration YAML.";
                statusTone = Sakura::CodeEditor::StatusTone::Info;
            },
            .onSubmit = [this](std::string value) {
                buffer = std::move(value);
                apply();
            },
        });
    }

    void open() {
        std::string yaml;
        if (Parser::YamlEncode(config.value, yaml) != Result::SUCCESS) {
            buffer.clear();
            appliedBuffer.clear();
            status = "Unable to serialize this block configuration.";
            statusTone = Sakura::CodeEditor::StatusTone::Error;
            return;
        }

        buffer = std::move(yaml);
        appliedBuffer = buffer;
        status = "Block configuration YAML.";
        statusTone = Sakura::CodeEditor::StatusTone::Info;
    }

    Sakura::FloatingLayer::Window window() {
        return {
            .id = config.id,
            .title = "Inspect & Edit (" + config.name + ")",
            .content = &editor,
            .size = {640.0f, 520.0f},
            .onClose = config.onClose,
        };
    }

 private:
    bool canApply() const {
        return buffer != appliedBuffer;
    }

    void apply() {
        if (!canApply() || !config.onApply) {
            return;
        }

        Parser::Map value;
        if (Parser::YamlDecode(buffer, value) != Result::SUCCESS) {
            status = "Invalid YAML. Fix the document before applying changes.";
            statusTone = Sakura::CodeEditor::StatusTone::Error;
            return;
        }

        appliedBuffer = buffer;
        status = "Changes applied.";
        statusTone = Sakura::CodeEditor::StatusTone::Success;
        config.onApply(std::move(value));
    }

    Config config;
    std::string buffer;
    std::string appliedBuffer;
    std::string status;
    Sakura::CodeEditor::StatusTone statusTone =
        Sakura::CodeEditor::StatusTone::Info;
    Sakura::CodeEditor editor;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_INSPECT_HH
