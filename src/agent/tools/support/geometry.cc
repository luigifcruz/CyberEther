#include "base.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Agent::Detail {
namespace {

constexpr F32 kSizeTolerance = 1.0f;

void AnnotateNode(Json& node, const Layout& layout) {
    if (!node.contains("name") || !node["name"].is_string()) {
        return;
    }

    const auto found = layout.nodes.find(node["name"].get<std::string>());
    if (found == layout.nodes.end()) {
        return;
    }

    const auto& measured = found->second;
    node["layout_status"] = "measured";
    node["bounds"] = Geometry::Bounds(measured.bounds);

    if (!measured.content.has_value()) {
        return;
    }

    const auto& content = *measured.content;
    const bool clippedX = content.x > measured.size.x + kSizeTolerance;
    const bool clippedY = content.y > measured.size.y + kSizeTolerance;

    node["content_size"] = {
        {"width", std::ceil(content.x)},
        {"height", std::ceil(content.y)},
    };
    node["clipped"] = clippedX || clippedY;
}

}  // namespace

Json Geometry::Bounds(const Rect& value) {
    return {
        {"x", value.x},
        {"y", value.y},
        {"width", value.width},
        {"height", value.height},
    };
}

void Geometry::Annotate(Json& body, const Layout& layout) {
    if (body.is_object() && body.contains("layout_status")) {
        return AnnotateNode(body, layout);
    }

    if (!body.is_structured()) {
        return;
    }

    for (auto& item : body) {
        Geometry::Annotate(item, layout);
    }
}

void Geometry::Viewport(Json& body, const Layout& layout) {
    if (!layout.viewport.has_value() || body.contains("error")) {
        return;
    }

    const auto& [visible, toolbar] = *layout.viewport;
    const F32 top = toolbar.bottom() + (toolbar.y - visible.y);
    const F32 height = std::max(0.0f, visible.bottom() - top);
    const Rect open = {visible.x, top, visible.width, height};

    body["viewport"] = {
        {"visible", Geometry::Bounds(visible)},
        {"toolbar", Geometry::Bounds(toolbar)},
        {"open_area", Geometry::Bounds(open)},
    };
}

void Geometry::Overlaps(Json& body, const Layout& layout) {
    std::vector<std::string> names;
    for (const auto& [name, node] : layout.nodes) {
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());

    Json pairs = Json::array();
    for (U64 a = 0; a < names.size(); ++a) {
        const auto& first = layout.nodes.at(names[a]).bounds;

        for (U64 b = a + 1; b < names.size(); ++b) {
            const auto& second = layout.nodes.at(names[b]).bounds;

            const F32 width = std::min(first.right(), second.right()) - std::max(first.x, second.x);
            const F32 height = std::min(first.bottom(), second.bottom()) - std::max(first.y, second.y);
            if (width > kSizeTolerance && height > kSizeTolerance) {
                pairs.push_back(Json::array({names[a], names[b]}));
            }
        }
    }

    if (!pairs.empty()) {
        body["overlaps"] = std::move(pairs);
    }
}

Result Geometry::Save(const Graph& flowgraph, const std::string& name, const Flowgraph::Metadata::NodeLayout& layout) {
    Parser::Map stored;
    JST_CHECK(flowgraph->metadata().get("node", stored, name));

    Parser::Map geometry;
    JST_CHECK(layout.serialize(geometry));

    for (const auto& entry : geometry) {
        stored[entry.key] = entry.value;
    }

    return flowgraph->metadata().set("node", stored, name);
}

}  // namespace Jetstream::Agent::Detail
