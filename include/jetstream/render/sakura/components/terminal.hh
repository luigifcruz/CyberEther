#ifndef JETSTREAM_RENDER_SAKURA_TERMINAL_HH
#define JETSTREAM_RENDER_SAKURA_TERMINAL_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Terminal : public Component {
    enum class CursorShape : U8 {
        Block,
        Bar,
        Underline,
        Hollow,
    };

    enum class Gesture : U8 {
        Press,
        Drag,
        Release,
    };

    struct Span {
        U16 column = 0;
        U16 width = 1;
        std::string text;
        ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
        bool bold = false;
        bool underline = false;
        bool strikethrough = false;

        bool operator==(const Span&) const = default;
    };

    struct Background {
        U16 column = 0;
        U16 width = 1;
        ColorRGBA<F32> color = {0.0f, 0.0f, 0.0f, 1.0f};

        bool operator==(const Background&) const = default;
    };

    struct Range {
        U16 start = 0;
        U16 end = 0;

        bool operator==(const Range&) const = default;
    };

    struct Link {
        U16 column = 0;
        U16 width = 1;
        std::string uri;

        bool operator==(const Link&) const = default;
    };

    struct Row {
        std::vector<Span> spans;
        std::vector<Background> backgrounds;
        std::vector<Link> links;
        std::optional<Range> selection;

        bool operator==(const Row&) const = default;
    };

    struct Cursor {
        bool visible = false;
        bool blinking = false;
        U16 column = 0;
        U16 row = 0;
        U16 width = 1;
        CursorShape shape = CursorShape::Block;
        std::string text;
        ColorRGBA<F32> textColor = {1.0f, 1.0f, 1.0f, 1.0f};

        bool operator==(const Cursor&) const = default;
    };

    struct Screen {
        U16 columns = 0;
        U16 rows = 0;
        std::vector<Row> lines;
        Cursor cursor;
        ColorRGBA<F32> base = {0.0f, 0.0f, 0.0f, 0.0f};
        ColorRGBA<F32> background = {0.0f, 0.0f, 0.0f, 0.0f};
        ColorRGBA<F32> fill = {0.0f, 0.0f, 0.0f, 0.0f};
        std::string selection;
        U64 scrollTotal = 0;
        U64 scrollOffset = 0;
        bool mouseTracking = false;

        bool operator==(const Screen&) const = default;
    };

    struct Grid {
        U16 columns = 0;
        U16 rows = 0;
        F32 cellWidth = 0.0f;
        F32 cellHeight = 0.0f;

        bool operator==(const Grid&) const = default;
    };

    struct Selection {
        Gesture gesture = Gesture::Press;
        U16 column = 0;
        U16 row = 0;
        U8 clicks = 1;

        bool operator==(const Selection&) const = default;
    };

    struct Config {
        std::string id;
        const Screen* screen = nullptr;
        U64 revision = 0;
        U64 focusRequest = 0;
        Extent2D<F32> size = {Fill, Fill};
        F32 fontSize = Typography::FontSize;
        F32 padding = 0.0f;
        F32 rounding = 0.0f;
        U8 corners = Box::AllCorners;
        bool altSendsEscape = false;
        std::string cursorColorKey = "editor_cursor";
        std::string selectionColorKey = "editor_selection";
        std::string scrollbarTrackColorKey = "editor_scrollbar_track";
        std::string scrollbarThumbColorKey = "editor_scrollbar_thumb";
        Callback<void(Grid)> onResize;
        Callback<void(const KeyEvent&)> onKey;
        Callback<void(std::string)> onText;
        Callback<void(std::string)> onPaste;
        Callback<void(I32 rows, Extent2D<F32> position, KeyModifiers modifiers)> onScroll;
        Callback<void(U64 row)> onScrollTo;
        Callback<void(Selection)> onSelect;
        Callback<void(const MouseEvent&)> onPointer;
        Callback<void(bool)> onFocusChange;
        Callback<void(I32 step)> onZoom;
        Callback<void(std::string uri)> onOpenLink;

        bool operator==(const Config&) const = default;
    };

    Terminal();
    ~Terminal();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    bool keyEvent(const KeyEvent& event) override;
    bool textEvent(const TextEvent& event) override;
    void focusChanged(bool focused) override;
    bool editsText() const override;
    bool focusesOnClick() const override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TERMINAL_HH
