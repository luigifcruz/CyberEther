#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_ROOT_COMPONENT_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_ROOT_COMPONENT_HH

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/components/text.hh>

#include "../context.hh"
#include "access.hh"
#include "dock_drag.hh"
#include "helpers.hh"
#include "popup_layer.hh"
#include "../metrics.hh"

#include <algorithm>
#include <array>
#include <optional>
#include <vector>

namespace Jetstream::Sakura {

struct RootComponent : public Component {
    static constexpr F64 kPreviewDuration = 0.12;
    static constexpr F32 kPreviewRadius = kPanelRounding;
    static constexpr F32 kPreviewBorder = 2.0f;
    static constexpr F32 kGhostOffset = 14.0f;
    static constexpr F32 kGhostPadX = 12.0f;
    static constexpr F32 kGhostHeight = 28.0f;
    static constexpr F32 kGhostIconGap = 6.0f;

    struct PreviewMotion {
        Rect from;
        Rect to;
        Rect rect;
        F64 moveStart = 0.0;
        F32 alpha = 0.0f;
        F32 alphaFrom = 0.0f;
        F32 alphaTo = 0.0f;
        F64 fadeStart = 0.0;

        static F32 Progress(F64 now, F64 start) {
            return static_cast<F32>(std::clamp((now - start) / kPreviewDuration, 0.0, 1.0));
        }

        static F32 Ease(F32 t) {
            const F32 inverse = 1.0f - t;
            return 1.0f - inverse * inverse * inverse;
        }

        static F32 Mix(F32 a, F32 b, F32 t) {
            return a + (b - a) * t;
        }

        void advance(F64 now, std::optional<Rect> goal);

        bool settled() const {
            return alpha == alphaTo && rect == to;
        }
    };

    Box background;
    PopupLayer popups;
    Box preview;
    Box ghostFrame;
    Text ghostIcon;
    Text ghost;
    DockDrag drag;
    PreviewMotion motion;
    Extent2D<F32> dragPoint = {-1.0f, -1.0f};
    bool dragged = false;
    bool reoffer = false;
    std::vector<Layer*> layers;
    Component* captured = nullptr;
    std::weak_ptr<bool> capturedAlive;
    const Component* capturedTarget = nullptr;
    std::weak_ptr<bool> capturedTargetAlive;
    std::array<bool, 3> capturedButtons{};
    std::array<bool, 3> cancelledButtons{};
    struct ModalFocus {
        Component* modal = nullptr;
        std::weak_ptr<bool> alive;
        Component* previous = nullptr;
        std::weak_ptr<bool> previousAlive;
        Component* focused = nullptr;
        std::weak_ptr<bool> focusedAlive;
    };
    std::vector<ModalFocus> modalFocus;
    Component* keyboardModal = nullptr;
    Component* pendingFocus = nullptr;
    std::weak_ptr<bool> pendingFocusAlive;
    bool preparingInput = false;

    RootComponent() {
        setClipsChildren(true);
    }

    void setLayers(std::vector<Layer*> next);

    void cancelCapture(bool notify = true);

    Component* activeModal() const;

    bool inModal(const Component& target, const Component& modal) const {
        return Access::within(target, modal) || popups.ownedBy(target, modal);
    }

    bool acceptsInput(const Component& target) const override {
        const auto* modal = activeModal();
        return !modal || &target == &popups || inModal(target, *modal);
    }

    void prepareInput() override;

    Extent2D<F32> measure(const Context&, Extent2D<F32> available) override {
        return {FillLength(available.x), FillLength(available.y)};
    }

    void layout(const Context& ctx) override;

    void relayoutLayers();

    Component* target(const Extent2D<F32>& point);

    std::vector<Component*> ordered();

    static MouseEvent away(MouseEvent event) {
        event.position = {-1.0e9f, -1.0e9f};
        return event;
    }

    bool mouseEvent(const MouseEvent& event) override;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_ROOT_COMPONENT_HH
