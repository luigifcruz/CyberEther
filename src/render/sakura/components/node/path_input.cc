#include <jetstream/render/sakura/components/node/path_input.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/node/content_stack.hh>
#include <jetstream/render/sakura/components/node/text_input.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../../detail/helpers.hh"
#include "../../detail/text_metrics.hh"
#include "../../metrics.hh"
#include "pointer.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura {

struct NodePathInput::Impl {
    Config config;
    Box background;
    NodeTextInput input;
    Label icon;
    TextMetrics metrics;
    Pressable press;
    Rect buttonRect;
    Rect inputRect;
    Pointer pointer;
};

NodePathInput::NodePathInput() {
    this->impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->input);
    addChild(impl->icon);
}

NodePathInput::~NodePathInput() = default;

void NodePathInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->input.update({
        .id = impl->config.id + ":path",
        .value = impl->config.value,
        .hint = "Select file...",
        .submit = NodeTextInput::Submit::OnEnter,
        .onChange = impl->config.onChange,
    });
}

Extent2D<F32> NodePathInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, NodeFrameHeight(ctx)), available.y)};
}

void NodePathInput::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const Rect clipRect = Intersect(bounds, clip());
    impl->metrics.setWindow(ctx.render);

    const F32 frameHeight = Px(ctx, NodeFrameHeight(ctx));
    const F32 buttonWidth = frameHeight;
    const F32 connector = Px(ctx, NodeContentStack::Spacing);
    const F32 inputWidth = std::max(0.0f, bounds.width - buttonWidth - connector);
    impl->inputRect = {bounds.x, bounds.y, inputWidth, frameHeight};
    impl->buttonRect = {bounds.x + inputWidth + connector, bounds.y, buttonWidth, frameHeight};

    impl->background.update({
        .instances = {
            {.rect = Snap({bounds.x, bounds.y, inputWidth + connector + buttonWidth, frameHeight}),
             .visible = !bounds.empty(), .backgroundColor = ctx.color("card")},
        },
        .clip = bounds,
        .cornerRadius = Px(ctx, kFrameRounding),
    });

    const F32 iconSize = Px(ctx, NodeFontOf(ctx)) * 0.7f;
    const F32 iconWidth = impl->metrics.measure(Typography::IconFont, ICON_LC_FOLDER_OPEN, iconSize);
    impl->icon.update({
        .instances = {{
            .rect = {impl->buttonRect.x + (impl->buttonRect.width - iconWidth) * 0.5f, impl->buttonRect.y,
                     iconWidth, impl->buttonRect.height},
            .str = ICON_LC_FOLDER_OPEN,
            .visible = !bounds.empty(),
            .color = ctx.color("text_primary"),
            .fontSize = iconSize,
            .alignment = {0, 1},
        }},
        .clip = bounds,
        .fontName = Typography::IconFont,
        .maxCharacters = 4,
    });

    layoutChild(ctx, impl->background, bounds);
    layoutChild(ctx, impl->input, impl->inputRect);
    layoutChild(ctx, impl->icon, bounds);

    const bool buttonHovered = impl->pointer.inside(Intersect(impl->buttonRect, clipRect));
    const bool inputHovered = impl->pointer.inside(Intersect(impl->inputRect, clipRect));
    impl->pointer.following = false;
    if (!config.value.empty() && (buttonHovered || inputHovered) && config.tooltip &&
        !impl->pointer.tooltip(ctx, bounds, config.tooltip())) {
        requestFrame();
    }
}

bool NodePathInput::mouseEvent(const MouseEvent& event) {
    const bool tooltip = impl->config.tooltip && !impl->config.value.empty();
    const bool moved = impl->pointer.track(event, Intersect(frame(), clip()), {impl->buttonRect, impl->inputRect});
    if (moved && tooltip) {
        invalidate(Dirty::Layout);
    }
    if (mouseEventChildren(event)) {
        return true;
    }
    if (impl->press.click(event, impl->buttonRect)) {
        return true;
    }
    if (impl->press.release(event, impl->buttonRect)) {
        if (impl->config.onBrowse) {
            impl->config.onBrowse();
        }
        return true;
    }
    return false;
}

}  // namespace Jetstream::Sakura
