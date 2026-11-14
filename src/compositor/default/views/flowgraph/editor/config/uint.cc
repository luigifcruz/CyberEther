#include "uint.hh"

namespace Jetstream {

Result FlowgraphConfigUIntField::update(Config config) {
    this->config = std::move(config);
    if (this->config.format != parsedFormat) {
        unit = Parser::Get<std::string>(this->config.format, "unit");
        parsedFormat = this->config.format;
    }
    value = 0;
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
        .unit = unit,
        .onChange = [this](U64 nextValue) {
            Parser::Map patch;
            patch[this->config.name] = nextValue;
            if (this->config.onApply) {
                this->config.onApply(std::move(patch), false, {});
            }
        },
    });
    return Result::SUCCESS;
}

Sakura::Component& FlowgraphConfigUIntField::component() {
    return frame;
}

}  // namespace Jetstream
