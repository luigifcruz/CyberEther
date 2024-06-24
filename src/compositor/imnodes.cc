#include "jetstream/compositor.hh"
#include "jetstream/instance.hh"

namespace Jetstream {

void Compositor::ImNodesStyleSetup() {
    auto &colors = ImNodes::GetStyle().Colors;
    colors[ImNodesCol_NodeBackground]         = IM_COL32(30, 30, 30, 255);
    colors[ImNodesCol_NodeBackgroundHovered]  = IM_COL32(30, 30, 30, 255);
    colors[ImNodesCol_NodeBackgroundSelected] = IM_COL32(35, 35, 35, 255);
    colors[ImNodesCol_NodeOutline]            = IM_COL32(20, 20, 20, 255);
    colors[ImNodesCol_Link]                   = IM_COL32(75, 75, 75, 255);
    colors[ImNodesCol_LinkHovered]            = IM_COL32(75, 75, 75, 255);
    colors[ImNodesCol_LinkSelected]           = IM_COL32(75, 75, 75, 255);
}

void Compositor::ImNodesStyleScale() {
    const auto& scalingFactor = instance.window().scalingFactor();
    auto& style = ImNodes::GetStyle();
    style.NodePadding               = ImVec2(4.0f * scalingFactor, 4.0f * scalingFactor);
    style.PinCircleRadius           = 2.0f  * scalingFactor;
    style.GridSpacing               = 20.0f * scalingFactor;
    style.NodeBorderThickness       = 0.5f  * scalingFactor;
    style.NodeCornerRounding        = 2.0f  * scalingFactor;
    style.LinkThickness             = 1.5f  * scalingFactor;
    style.PinLineThickness          = 0.5f  * scalingFactor;
    style.LinkLineSegmentsPerLength = 0.2f  / scalingFactor;
}

}  // namespace Jetstream