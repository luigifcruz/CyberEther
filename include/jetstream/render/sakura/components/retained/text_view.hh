#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_VIEW_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_VIEW_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/text_grid.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct TextView : public Component {
    struct Config {
        std::string id;
        std::string value;
        F32 fontSize = 15.0f;
        F32 fontScale = 1.0f;
        std::string fontName = "default_mono";
        bool monospace = true;
        bool lineNumbers = false;
        bool stickToBottom = false;
        bool scrollbar = true;
        TextGrid::Wrap wrap = TextGrid::Wrap::None;
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
        std::function<const std::vector<std::vector<TextGrid::StyleId>>&(
            const std::vector<std::string>& lines, U64 revision)> styler;
        std::function<bool(TextGrid::Position)> onPositionClick;
        std::function<bool(TextGrid::Position)> isPositionInteractive;
    };

    TextView();
    ~TextView();

    bool update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_VIEW_HH
