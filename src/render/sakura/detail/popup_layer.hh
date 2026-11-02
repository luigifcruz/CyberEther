#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_POPUP_LAYER_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_POPUP_LAYER_HH

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/components/popup.hh>

#include "../context.hh"

#include <functional>
#include <array>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace Jetstream::Sakura {

struct PopupRequest {
    Component* content = nullptr;
    Rect anchor;
    PopupPlacement placement = PopupPlacement::Below;
    bool panel = false;
    F32 padding = 12.0f;
    F32 maxWidth = 0.0f;
    Extent2D<F32> size = {0.0f, 0.0f};
    bool interactive = true;
    bool delayed = false;
    Callback<void()> onDismiss;
    std::optional<Rect> passthrough;

    bool operator==(const PopupRequest&) const = default;
};

struct PopupLayer : public Layer {
    PopupLayer();
    ~PopupLayer();

    void replace(Component& owner, std::weak_ptr<bool> alive, std::vector<PopupRequest> requests);
    bool covers(const Extent2D<F32>& point) const override;
    bool interactiveAt(const Extent2D<F32>& point) const;
    bool ownedBy(const Component& content, const Component& owner) const;
    void prepareInput() override;
    bool capturesInput() const;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    void wentOnscreen(U64 frame) override;

 private:
    struct Entry {
        Component* owner = nullptr;
        std::weak_ptr<bool> alive;
        std::vector<PopupRequest> requests;
        std::vector<std::weak_ptr<bool>> contents;

        void drop(U64 index) {
            requests.erase(requests.begin() + static_cast<I64>(index));
            contents.erase(contents.begin() + static_cast<I64>(index));
        }
    };

    struct Placed {
        Component* content = nullptr;
        std::weak_ptr<bool> alive;
        Rect rect;
    };

    struct Claim {
        const Entry* entry = nullptr;
        const PopupRequest* request = nullptr;
    };

    static constexpr U64 kArrangePasses = 32;

    void expire();
    void prune();
    void pruneFrames();
    void changed();
    void withdraw();
    void arrange(const Context& ctx);
    Claim claim(const Component* content) const;
    bool route(const MouseEvent& event);

    std::vector<Entry> entries;
    U64 revision = 0;
    bool arranging = false;
    Component* captured = nullptr;
    std::weak_ptr<bool> capturedAlive;
    std::array<bool, 3> capturedButtons{};
    std::vector<Placed> placed;
    std::unordered_map<const Component*, F64> requestedSince;
    std::unordered_map<const Component*, std::unique_ptr<Box>> frames;
    std::unordered_map<const Component*, std::weak_ptr<bool>> frameContents;
};

void RequestPopup(const Context& ctx, PopupRequest request);
Component* LayoutOwner();

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_POPUP_LAYER_HH
