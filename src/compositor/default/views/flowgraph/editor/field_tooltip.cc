#include "field_tooltip.hh"

namespace Jetstream {

void FlowgraphFieldTooltip::update(const std::string& id, const std::string& text, F32 wrapWidth) {
    if (text != str) {
        str = text;
        ++revision;
    }
    if (!tooltip.requested() || str.empty()) {
        return;
    }
    label.update({
        .str = str,
        .wrapped = true,
    });
    tooltip.update({
        .id = id + "HelpTooltip",
        .content = &label,
        .wrapWidth = wrapWidth,
    });
}

Sakura::NodeOverlay FlowgraphFieldTooltip::overlay() {
    if (str.empty()) {
        return {};
    }
    return {
        .content = [this]() {
            return tooltip.show();
        },
        .revision = revision,
    };
}

}  // namespace Jetstream
