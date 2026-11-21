#include "python.hh"

namespace Jetstream {

FlowgraphNodeHeightSpec FlowgraphConfigPythonField::heightSpec() const {
    return {
        .policy = FlowgraphNodeHeightPolicy::FillRemaining,
        .minimum = 120.0f,
    };
}

Result FlowgraphConfigPythonField::update(Config config) {
    this->config = std::move(config);
    std::string nextValue;
    JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, nextValue));
    parsedEncoded = std::move(nextValue);
    const std::string& shown = this->config.draft.value_or(parsedEncoded);
    frame.update({
        .id = this->config.id,
        .label = this->config.label,
        .title = false,
        .background = false,
        .content = &editor,
    });
    const bool dirty = shown != parsedEncoded;
    editor.update({
        .id = this->config.id + "Editor",
        .value = shown,
        .consoleOutput = this->config.consoleOutput,
        .status = dirty ? jst::fmt::format("Edited. Press {}+Enter to run.", Sakura::Shortcuts::PrimaryName())
                        : this->config.status,
        .height = 0.0f,
        .statusTone = dirty ? Sakura::StatusTone::Info : this->config.statusTone,
        .consoleVisible = this->config.consoleVisible,
        .maxAutoHeightWindowRatio = 0.65f,
        .language = Sakura::CodeEditor::Language::Python,
        .lineNumbers = true,
        .lineWrapping = false,
        .fontSize = 15.0f,
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
        .onContentOverflow = this->config.onContentOverflow,
    });
    return Result::SUCCESS;
}

Sakura::Component& FlowgraphConfigPythonField::component() {
    return frame;
}

}  // namespace Jetstream
