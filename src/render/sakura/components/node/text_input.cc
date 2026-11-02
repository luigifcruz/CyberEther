#include <jetstream/render/sakura/components/node/text_input.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/text_grid.hh>

#include "../../detail/helpers.hh"
#include "../../detail/text_metrics.hh"
#include "../../metrics.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Sakura {

struct NodeTextInput::Impl {
    Config config;
    Box frame;
    TextGrid grid;
    Label overlay;
    TextMetrics metrics;
    std::string buffer;
    bool submitted = false;
    bool cancelled = false;

    void focusChanged(bool focused) {
        if (focused) {
            return;
        }
        if (!submitted && !cancelled && config.submit == Submit::OnCommit && buffer != config.value &&
            config.onChange) {
            config.onChange(buffer);
        }
        grid.revert();
        buffer = config.value;
        submitted = false;
        cancelled = false;
        if (config.onDeactivate) {
            config.onDeactivate();
        }
    }
};

NodeTextInput::NodeTextInput() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->frame);
    addChild(impl->grid);
    addChild(impl->overlay);
}

NodeTextInput::~NodeTextInput() = default;

void NodeTextInput::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (!impl->grid.focused()) {
        impl->buffer = impl->config.value;
    }
}

bool NodeTextInput::focused() const {
    return impl->grid.focused();
}

void NodeTextInput::focus(bool selectAll) {
    impl->grid.focus(selectAll);
}

void NodeTextInput::focusWhenPlaced(bool selectAll) {
    impl->grid.offerFocus(true, selectAll);
}

Extent2D<F32> NodeTextInput::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), ResolveLength(ctx, impl->config.height, Px(ctx, NodeFrameHeight(ctx)), available.y)};
}

void NodeTextInput::layout(const Context& ctx) {
    auto& config = impl->config;
    const Rect bounds = frame();

    const bool focused = impl->grid.focused();

    impl->metrics.setWindow(ctx.render);

    const F32 fontSize = Px(ctx, NodeFontOf(ctx));
    const F32 padX = Px(ctx, kNodeFramePadX);
    const F32 padY = std::max(0.0f, (bounds.height - fontSize) * 0.5f);

    impl->frame.update({
        .instances = {{
            .rect = Snap(bounds),
            .visible = config.background && !bounds.empty(),
            .backgroundColor = ctx.color("card"),
        }},
        .clip = bounds,
        .cornerRadius = Px(ctx, kFrameRounding),
    });

    impl->grid.update({
        .id = config.id + ":grid",
        .value = focused ? impl->buffer : config.value,
        .editable = true,
        .fontSize = fontSize,
        .lineHeight = 1.0f,
        .fontName = Typography::MonoFont,
        .monospace = true,
        .lineNumbers = false,
        .showActiveLine = false,
        .scrollbar = false,
        .wrap = TextGrid::Wrap::None,
        .padding = Padding{padX, padY, padX, padY},
        .visibleLineCapacity = 4,
        .backgroundColorKey = "transparent",
        .textColorKey = config.textColorKey,
        .selectionColorKey = "text_selected_bg",
        .selectionMatchColorKey = "transparent",
        .activeLineColorKey = "transparent",
        .cursorColorKey = "text_primary",
        .submitOnEnter = true,
        .singleLine = true,
        .selectAllOnFocus = config.selectAllOnFocus,
        .characterFilter = [this](U32 character) {
            return !impl->config.characterFilter || impl->config.characterFilter(character);
        },
        .onChange = [this](std::string value) {
            impl->buffer = std::move(value);
        },
        .onSubmit = [this](std::string value) {
            impl->buffer = std::move(value);
            impl->submitted = true;
            if (impl->config.onChange) {
                impl->config.onChange(impl->buffer);
            }
            impl->grid.blur();
        },
        .onCancel = [this]() {
            impl->cancelled = true;
        },
        .onFocusChange = [this](bool focused) {
            impl->focusChanged(focused);
        },
    });

    const F32 unitWidth = impl->metrics.measure(Typography::MonoFont, config.unit, fontSize);
    const Rect lineRect = {bounds.x + padX, bounds.y + padY, std::max(0.0f, bounds.width - 2.0f * padX), fontSize};
    const Rect unitRect = {bounds.right() - Px(ctx, kNodeUnitInset) - unitWidth, bounds.y, unitWidth, bounds.height};
    impl->overlay.update({
        .instances = {
            {
                .rect = lineRect,
                .str = config.hint,
                .visible = impl->buffer.empty() && !config.hint.empty() && !bounds.empty(),
                .color = ctx.color("text_secondary"),
                .fontSize = fontSize,
                .alignment = {0, 1},
            },
            {
                .rect = unitRect,
                .str = config.unit,
                .visible = !config.unit.empty() && !bounds.empty(),
                .color = ctx.color("text_secondary"),
                .fontSize = fontSize,
                .alignment = {0, 1},
            },
        },
        .clip = bounds,
        .fontName = Typography::MonoFont,
        .maxCharacters = 128,
    });

    layoutChild(ctx, impl->frame, bounds);
    layoutChild(ctx, impl->grid, bounds);
    layoutChild(ctx, impl->overlay, bounds);
}

bool NodeTextInput::mouseEvent(const MouseEvent& event) {
    return mouseEventChildren(event);
}

}  // namespace Jetstream::Sakura
