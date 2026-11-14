#include "markdown.hh"

namespace Jetstream {

FlowgraphNodeHeightSpec FlowgraphConfigMarkdownField::heightSpec() const {
    return {
        .policy = FlowgraphNodeHeightPolicy::FillRemaining,
        .minimum = 120.0f,
    };
}

Result FlowgraphConfigMarkdownField::update(Config config) {
    this->config = std::move(config);
    JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
    typed.reset();
    frame.update({
        .id = this->config.id,
        .label = this->config.label,
        .title = false,
        .background = false,
        .content = &markdownField,
    });
    markdownField.update({
        .id = this->config.id + "Markdown",
        .value = editing ? this->config.draft.value_or(value) : this->config.preview.value_or(value),
        .editing = editing,
        .onChange = [this](std::string nextValue) {
            typed = nextValue;
            if (this->config.onDraft) {
                this->config.onDraft(std::move(nextValue), value);
            }
        },
        .onEdit = [this]() {
            editing = true;
        },
        .onDone = [this]() {
            Parser::Map patch;
            patch[this->config.name] = typed.value_or(this->config.draft.value_or(value));
            if (this->config.onApply) {
                this->config.onApply(std::move(patch), false, {});
            }
            editing = false;
        },
        .onContentOverflow = this->config.onContentOverflow,
        .onLink = [this](const std::string& url) {
            if (!this->config.onOpenUrl) {
                return false;
            }
            this->config.onOpenUrl(url);
            return true;
        },
    });
    return Result::SUCCESS;
}

Sakura::Component& FlowgraphConfigMarkdownField::component() {
    return frame;
}

}  // namespace Jetstream
