#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_MARKDOWN_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_MARKDOWN_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/text_grid.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct TextMarkdown : public Component {
    static constexpr const char* BodyFont = Typography::BodyFont;

    using StyleId = U8;

    struct Style {
        static constexpr StyleId Plain = 0;
        static constexpr StyleId Bold = 1;
        static constexpr StyleId Italic = 2;
        static constexpr StyleId BoldItalic = 3;
        static constexpr StyleId Code = 4;
        static constexpr StyleId Link = 5;
        static constexpr StyleId CodeBlock = 6;
        static constexpr StyleId Count = 6;
        static constexpr StyleId CalloutBase = 7;
        static constexpr StyleId CalloutVariants = 4;
        static constexpr StyleId CalloutTones = 6;

        static constexpr StyleId Callout(StyleId tone, StyleId emphasis) {
            return static_cast<StyleId>(CalloutBase + tone * CalloutVariants + emphasis);
        }
    };

    struct Config {
        std::string id;
        std::string value;
        F32 fontSize = Typography::FontSize;
        bool scrollbar = false;
        std::optional<Padding> padding;
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
        std::string codeBlockColorKey = "markdown_code_block";
        std::string codeBlockBorderColorKey = "markdown_code_block_border";
        std::vector<std::string> styleColorKeys = {"", "", "", "", "cyber_blue", ""};
        std::vector<std::string> styleFonts = {"default_body_bold", "default_body_italic",
                                               "default_body_bold_italic", "default_mono", "", "default_mono"};
        std::vector<std::string> styleBackgroundColorKeys = {"", "", "", "editor_scrollbar_track", "", ""};
        std::vector<F32> styleScales = {1.0f, 1.0f, 1.0f, 0.9f, 1.0f, 0.9f};
    };

    TextMarkdown();
    ~TextMarkdown();

    bool update(Config config);
    F32 naturalWidth() const;
    const TextGrid::Metrics& metrics() const;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_MARKDOWN_HH
