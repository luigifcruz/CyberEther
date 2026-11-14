#include "inspect.hh"

namespace Jetstream {

void FlowgraphNodeInspector::update(Config config) {
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

void FlowgraphNodeInspector::open() {
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

Sakura::FloatingLayer::Window FlowgraphNodeInspector::window() {
    return {
        .id = config.id,
        .title = jst::fmt::format("Inspect & Edit ({})", config.name),
        .content = &editor,
        .size = {640.0f, 520.0f},
        .onClose = config.onClose,
    };
}

bool FlowgraphNodeInspector::canApply() const {
    return buffer != appliedBuffer;
}

void FlowgraphNodeInspector::apply() {
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

}  // namespace Jetstream
