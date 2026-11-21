#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_FIELD_TOOLTIP_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_FIELD_TOOLTIP_HH

#include "../../components/tooltip.hh"
#include "jetstream/render/sakura/base.hh"

#include "jetstream/types.hh"

#include <string>

namespace Jetstream {

struct FlowgraphFieldTooltip {
    void update(const std::string& id, const std::string& text, F32 wrapWidth = 420.0f);
    Sakura::NodeOverlay overlay();

 private:
    std::string str;
    U64 revision = 0;
    Tooltip tooltip;
    Sakura::Text label;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_FIELD_TOOLTIP_HH
