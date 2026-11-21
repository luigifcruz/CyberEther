#include "text.hh"

namespace Jetstream {

Result FlowgraphConfigTextField::update(Config config) {
    this->config = std::move(config);
    value.clear();
    JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
    help.update(this->config.id, this->config.help);
    frame.update({
        .id = this->config.id,
        .label = this->config.label,
        .tooltip = help.overlay(),
        .content = &input,
    });
    input.update({
        .id = this->config.id + "Input",
        .value = value,
        .submit = Sakura::NodeTextInput::Submit::OnEnter,
        .onChange = [this](const std::string& nextValue) {
            Parser::Map patch;
            patch[this->config.name] = nextValue;
            if (this->config.onApply) {
                this->config.onApply(std::move(patch), false, {});
            }
        },
    });
    return Result::SUCCESS;
}

Sakura::Component& FlowgraphConfigTextField::component() {
    return frame;
}

}  // namespace Jetstream
