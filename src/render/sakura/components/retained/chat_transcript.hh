#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_TRANSCRIPT_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_TRANSCRIPT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/box.hh>
#include "../../retained/shapes.hh"
#include <jetstream/render/sakura/components/retained/button.hh>
#include <jetstream/render/sakura/components/retained/chat.hh>
#include <jetstream/render/sakura/components/retained/scroll_view.hh>
#include <jetstream/render/sakura/components/retained/text_markdown.hh>
#include <jetstream/render/sakura/components/retained/text_view.hh>
#include <jetstream/types.hh>

#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace Jetstream::Render {
class Window;
}  // namespace Jetstream::Render

namespace Jetstream::Sakura::Retained {

struct ChatMessageList : public Component {
    explicit ChatMessageList(const Chat::Config& config);

    void reset();
    void resetScroll();
    void stickToTail();

 protected:
    void layout(const Context& ctx) override;

 private:
    enum class BlockKind : U8 {
        User,
        Markdown,
        Disclosure,
    };

    struct Metrics {
        Rect bounds;
        Rect clip;
        bool visible = false;
        F32 unit = 1.0f;
        F32 font = 0.0f;
        F32 lineHeight = 0.0f;
        F32 textPad = 0.0f;
        F32 scrollbarReserve = 0.0f;
        F32 contentWidth = 0.0f;
        F32 maxBubbleWidth = 0.0f;
        F32 maxUserTextWidth = 0.0f;
        F32 partGap = 0.0f;
        F32 summaryInset = 0.0f;
        F32 caretInset = 0.0f;
        CaretSize caret;
        F32 labelInset = 0.0f;
        F32 visibleTop = 0.0f;
        F32 visibleBottom = 0.0f;
        const Render::Window* window = nullptr;

        bool inViewport(const Rect& rect) const;
        bool inBounds(const Rect& rect) const;
    };

    struct BlockKey {
        U64 generation = 0;
        U64 message = 0;
        U64 part = 0;
        BlockKind kind = BlockKind::Markdown;

        bool operator==(const BlockKey&) const = default;

        bool operator<(const BlockKey& other) const {
            return std::tie(generation, message, part, kind) <
                   std::tie(other.generation, other.message, other.part, other.kind);
        }
    };

    struct Block {
        BlockKey key;
        U64 revision = 0;
        std::string value;
        std::string summary;
        std::string toggleLabel;
        bool expanded = false;
        bool busy = false;
        F32 width = 0.0f;
        F32 height = 0.0f;
        F32 bodyHeight = 0.0f;
        F32 summaryHeight = 0.0f;
        Extent2D<F32> toggleSize = {0.0f, 0.0f};
    };

    struct Measurement {
        U64 revision = std::numeric_limits<U64>::max();
        F32 width = -1.0f;
        F32 fontSize = -1.0f;
        F32 unit = -1.0f;
        const Render::Window* window = nullptr;
        bool expanded = false;
        bool busy = false;
        F32 measuredWidth = 0.0f;
        F32 height = 0.0f;
        F32 bodyHeight = 0.0f;
        F32 summaryHeight = 0.0f;
        Extent2D<F32> toggleSize = {0.0f, 0.0f};

        bool matches(const Block& block, F32 measuredAt, const Metrics& m) const;
        void store(const Block& block, F32 measuredAt, const Metrics& m);
    };

    struct RowContext {
        U64 generation = 0;
        F32 contentWidth = 0.0f;
        F32 unit = 0.0f;
        const Render::Window* window = nullptr;
        U64 disclosureRevision = 0;

        bool operator==(const RowContext&) const = default;
    };

    struct UserSlot {
        TextView view;
        std::optional<BlockKey> key;
        bool identityChanged = false;
        bool activeLastFrame = false;

        auto parts() { return std::tie(view); }
    };

    struct MarkdownSlot {
        TextMarkdown view;
        std::optional<BlockKey> key;
        bool identityChanged = false;
        bool activeLastFrame = false;

        auto parts() { return std::tie(view); }
    };

    struct DisclosureSlot {
        Button toggle;
        Box caret;
        TextView summary;
        Box shimmer;
        std::optional<BlockKey> key;
        bool identityChanged = false;
        bool activeLastFrame = false;

        auto parts() { return std::tie(toggle, caret, summary, shimmer); }
    };

    struct UserPlacement {
        Block* block = nullptr;
        Rect bubble;
        Rect text;
    };

    struct MarkdownPlacement {
        Block* block = nullptr;
        Rect rect;
    };

    struct DisclosurePlacement {
        Block* block = nullptr;
        Rect toggle;
        Rect caret;
        Rect summary;
    };

    struct Placements {
        std::vector<UserPlacement> users;
        std::vector<MarkdownPlacement> markdown;
        std::vector<DisclosurePlacement> disclosures;
    };

    const Chat::Config& config;
    Box bubbleBoxes;
    ScrollView scroll;
    // Measurers stay off-tree so history uses CPU state, not GPU drawables.
    TextView userMeasurer;
    TextMarkdown markdownMeasurer;
    Button toggleMeasurer;
    TextView summaryMeasurer;
    std::map<BlockKey, Measurement> measurements;
    std::vector<std::vector<Block>> messageBlocks;
    std::vector<F32> rowWidth;
    std::vector<F32> rowHeight;
    std::vector<std::shared_ptr<const Chat::Message>> rowMessages;
    RowContext rowContext;
    std::vector<std::unique_ptr<UserSlot>> userSlots;
    std::vector<std::unique_ptr<MarkdownSlot>> markdownSlots;
    std::vector<std::unique_ptr<DisclosureSlot>> disclosureSlots;
    std::chrono::steady_clock::time_point epoch = std::chrono::steady_clock::now();

    F32 scrollY = 0.0f;
    bool followTail = true;
    F32 lastMaxScroll = 0.0f;
    std::map<std::pair<U64, U64>, bool> expansions;
    U64 disclosureRevision = 0;

    template<typename Slot>
    void ensureSlots(std::vector<std::unique_ptr<Slot>>& slots, U64 count) {
        while (slots.size() < count) {
            auto& slot = *slots.emplace_back(std::make_unique<Slot>());
            std::apply([this](auto&... parts) { (add(parts), ...); }, slot.parts());
        }
    }

    void toggleDisclosure(U64 expectedGeneration, U64 messageIndex, U64 partKey);
    bool expanded(U64 message, U64 part) const;

    Metrics metricsFor(const Context& ctx) const;
    std::string scopedId(std::string_view suffix) const;
    std::string slotId(std::string_view element, U64 index) const;
    std::string blockId(const BlockKey& key, std::string_view element) const;
    Block makeBlock(U64 message, U64 part, BlockKind kind) const;

    void resetCaches(const Metrics& m);
    void buildRows(const Context& ctx, const Metrics& m);
    void buildUserRow(const Context& ctx, const Metrics& m, U64 index);
    void buildAssistantRow(const Context& ctx, const Metrics& m, U64 index);
    void addMarkdownBlock(const Context& ctx, const Metrics& m, U64 index, const Chat::Part& part);
    void addDisclosureBlock(const Context& ctx, const Metrics& m, U64 index, const Chat::Part& part);
    void measureDisclosure(const Context& ctx, const Metrics& m, Block& block);

    F32 gapAfter(const Metrics& m, U64 index) const;
    F32 contentHeight(const Metrics& m) const;
    void clampScroll(const Metrics& m, F32 height);

    Placements place(const Metrics& m);
    void placeUser(const Metrics& m, U64 index, F32 y, Placements& placements);
    void placeAssistant(const Metrics& m, U64 index, F32 y, Placements& placements);
    F32 placeDisclosure(const Metrics& m, Block& block, F32 width, F32 y, Placements& placements);
    U64 assignableDisclosures(const std::vector<DisclosurePlacement>& placements) const;

    std::vector<Box::Instance> bindUserSlots(const Context& ctx, const Metrics& m,
                                             const std::vector<const UserPlacement*>& bySlot);
    void bindMarkdownSlots(const Context& ctx, const Metrics& m,
                           const std::vector<const MarkdownPlacement*>& bySlot);
    void bindDisclosureSlots(const Context& ctx, const Metrics& m,
                             const std::vector<const DisclosurePlacement*>& bySlot);
    void bindIdleDisclosureSlot(const Context& ctx, const Metrics& m, DisclosureSlot& slot, U64 index);
    void bindDisclosureSlot(const Context& ctx, const Metrics& m, DisclosureSlot& slot, U64 index,
                            const DisclosurePlacement& placement);
    void bindBubbles(const Context& ctx, const Metrics& m, std::vector<Box::Instance> bubbles);
    void bindScroll(const Context& ctx, const Metrics& m, F32 height);

    static TextView::Config userTextConfig(std::string id, std::string value, F32 font);
    static TextView::Config summaryTextConfig(std::string id, std::string value, F32 font);
    static TextMarkdown::Config markdownConfig(std::string id, std::string value, F32 font, bool disclosure);
    static Button::Config toggleConfig(std::string id, std::string label, const Metrics& m, bool busy);
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_TRANSCRIPT_HH
