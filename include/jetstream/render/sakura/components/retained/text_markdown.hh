#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_MARKDOWN_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_MARKDOWN_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct TextMarkdown : public Component {
    struct Config {
        std::string id;
        std::string value;
        F32 fontSize = 15.0f;
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
        std::vector<std::string> styleColorKeys = {"", "", "", "", "cyber_blue"};
        std::vector<std::string> styleFonts = {"default_body_bold", "default_body_italic",
                                               "default_body_bold_italic", "default_mono", ""};
        std::vector<std::string> styleBackgroundColorKeys;
    };

    TextMarkdown();
    ~TextMarkdown();

    bool update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_MARKDOWN_HH
