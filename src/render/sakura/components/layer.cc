#include <jetstream/render/sakura/components/layer.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <vector>

namespace Jetstream::Sakura {

Layer::Layer() = default;

Layer::~Layer() = default;

void Layer::update(Config config) {
    auto* next = config.content;
    if (next == content && (!next || hasChild(*next))) {
        return;
    }
    content = next;
    invalidate(Dirty::Layout);
    std::vector<Component*> children;
    if (next) {
        children.push_back(next);
    }
    mountChildren(children);
}

bool Layer::covers(const Extent2D<F32>&) const {
    return content != nullptr;
}

bool Layer::modal() const {
    return false;
}

bool Layer::occludes(const Extent2D<F32>& point) const {
    return covers(point);
}

Extent2D<F32> Layer::measure(const Context&, Extent2D<F32> available) {
    return {FillLength(available.x), FillLength(available.y)};
}

void Layer::layout(const Context& ctx) {
    if (content) {
        layoutChild(ctx, *content, frame());
    }
}

}  // namespace Jetstream::Sakura
