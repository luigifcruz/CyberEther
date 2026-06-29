#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_GRID_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_GRID_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct TextGrid : public Component {
    struct Position {
        U64 line = 0;
        U64 column = 0;

        bool operator==(const Position&) const = default;
    };

    using StyleId = U8;

    enum class Wrap : U8 {
        None,
        Character,
        Word,
    };

    struct LineMetrics {
        F32 top = 0.0f;
        F32 height = 0.0f;
    };

    struct Metrics {
        F32 contentHeight = 0.0f;
        std::vector<LineMetrics> sourceLines;
    };

    struct Config {
        std::string id;
        std::string value;
        bool editable = false;
        F32 fontSize = 15.0f;
        F32 fontScale = 1.0f;
        std::string fontName = "default_mono";
        bool monospace = true;
        bool lineNumbers = false;
        bool showActiveLine = true;
        bool stickToBottom = false;
        bool scrollbar = true;
        Wrap wrap = Wrap::None;
        std::vector<F32> lineScale;
        std::vector<F32> lineTopGap;
        std::vector<F32> lineIndent;
        std::string backgroundColorKey = "transparent";
        std::string textColorKey = "text_primary";
        std::string lineNumberColorKey = "editor_line_number";
        std::string gutterSeparatorColorKey = "editor_gutter_separator";
        std::string selectionColorKey = "editor_selection";
        std::string selectionMatchColorKey = "editor_selection_match";
        std::string activeLineColorKey = "editor_active_line";
        std::string cursorColorKey = "editor_cursor";
        std::string scrollbarTrackColorKey = "editor_scrollbar_track";
        std::string scrollbarThumbColorKey = "editor_scrollbar_thumb";
        std::vector<std::string> styleColorKeys;
        std::vector<std::string> styleFonts;
        std::vector<std::string> styleBackgroundColorKeys;
        U64 maxLineSegments = 64;
        bool submitOnEnter = false;
        std::function<const std::vector<std::vector<StyleId>>&(
            const std::vector<std::string>& lines, U64 revision)> styler;
        std::function<bool(StyleId)> isStyleCommentOrString;
        std::function<void(std::string)> onChange;
        std::function<void(std::string)> onSubmit;
        std::function<void(Position start, Position end)> onSelect;
        std::function<bool(Position)> onPositionClick;
        std::function<bool(Position)> isPositionInteractive;
        std::function<void(const Metrics&)> onLayout;
        std::function<std::optional<std::string>(
            const std::vector<std::string>& lines, Position cursor)> computeNewlineIndent;
        std::function<std::optional<std::pair<Position, Position>>(
            const std::vector<std::string>& lines, Position anchor, Position cursor)> expandSelection;
    };

    TextGrid();
    ~TextGrid();

    bool update(Config config);
    const Metrics& metrics() const;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool event(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_GRID_HH
