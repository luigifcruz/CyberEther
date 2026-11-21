#include "path.hh"

namespace Jetstream {

Result FlowgraphConfigPathField::update(Config config) {
    this->config = std::move(config);
    if (this->config.format != parsedFormat) {
        parseFormat();
    }
    value.clear();
    JST_CHECK(Parser::Deserialize(this->config.values, this->config.name, value));
    help.update(this->config.id, this->config.help);
    frame.update({
        .id = this->config.id,
        .label = this->config.label,
        .tooltip = help.overlay(),
        .content = &input,
    });
    pathTooltip.update(this->config.id + "Path", value);
    input.update({
        .id = this->config.id + "Path",
        .value = value,
        .tooltip = pathTooltip.overlay(),
        .onChange = [this](const std::string& nextValue) {
            Parser::Map patch;
            patch[this->config.name] = nextValue;
            if (this->config.onApply) {
                this->config.onApply(std::move(patch), false, {});
            }
        },
        .onBrowse = [this]() {
            browse();
        },
    });
    return Result::SUCCESS;
}

Sakura::Component& FlowgraphConfigPathField::component() {
    return frame;
}

void FlowgraphConfigPathField::parseFormat() {
    parsedFormat = config.format;
    mode = Parser::Get<std::string>(config.format, "type") == "filesave" ? Mode::Save : Mode::Open;
    extensions = Parser::Get<std::vector<std::string>>(config.format, "extensions");
}

void FlowgraphConfigPathField::browse() const {
    auto applyPath = [name = config.name,
                      onApply = config.onApply](std::string path) mutable {
        Parser::Map patch;
        patch[name] = std::move(path);
        if (onApply) {
            onApply(std::move(patch), false, {});
        }
    };

    if (config.onBrowsePath) {
        config.onBrowsePath(mode == Mode::Save, extensions, std::move(applyPath));
        return;
    }
}

}  // namespace Jetstream
