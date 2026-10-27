#ifndef JETSTREAM_RENDER_SAKURA_TEXT_MARKDOWN_HH
#define JETSTREAM_RENDER_SAKURA_TEXT_MARKDOWN_HH

#include <jetstream/render/sakura/callback.hh>
#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

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
        static constexpr StyleId SyntaxBase = CalloutBase + CalloutTones * CalloutVariants;
        static constexpr StyleId SyntaxStyles = 9;
        static constexpr StyleId StatLabel = SyntaxBase + SyntaxStyles;
        static constexpr StyleId FootnoteBase = StatLabel + 1;
        static constexpr StyleId FootnoteVariants = 4;
        static constexpr StyleId FootnoteRef = FootnoteBase + FootnoteVariants;
        static constexpr StyleId Muted = FootnoteRef + 1;

        static constexpr StyleId Callout(StyleId tone, StyleId emphasis) {
            return static_cast<StyleId>(CalloutBase + tone * CalloutVariants + emphasis);
        }

        static constexpr StyleId Syntax(StyleId style) {
            return static_cast<StyleId>(SyntaxBase + style - 1);
        }

        static constexpr StyleId Footnote(StyleId emphasis) {
            return static_cast<StyleId>(FootnoteBase + emphasis);
        }
    };

    struct Hero {
        std::string eyebrow;
        std::string title;
        std::string subtitle;

        bool operator==(const Hero&) const = default;
    };

    struct Config {
        std::string id;
        std::string value;
        F32 fontSize = Typography::FontSize;
        bool scrollbar = false;
        Extent2D<F32> scrollbarInset = {0.0f, 0.0f};
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
        std::vector<std::string> styleFonts = {Typography::BodyBoldFont, Typography::BodyItalicFont,
                                               Typography::BodyBoldItalicFont, Typography::MonoFont, "",
                                               Typography::MonoFont};
        std::vector<std::string> styleBackgroundColorKeys = {"", "", "", "editor_scrollbar_track", "", ""};
        std::vector<F32> styleScales = {1.0f, 1.0f, 1.0f, 0.9f, 1.0f, 1.0f};
        std::vector<std::string> syntaxColorKeys = {"editor_comment", "editor_keyword", "editor_string",
                                                    "editor_constant", "editor_function", "editor_type",
                                                    "editor_constant", "editor_keyword", "editor_type"};
        Callback<bool(const std::string&)> onLink;
        std::optional<Hero> hero;
        bool softBreaks = false;
        std::vector<bool> detailsOpen;
        Callback<void(const std::vector<bool>&)> onDetailsToggle;
        Callback<bool(F32 offset)> onReveal;

        bool operator==(const Config&) const = default;
    };

    struct Heading {
        std::string text;
        U64 level = 0;
        U64 line = 0;

        bool operator==(const Heading&) const = default;
    };

    TextMarkdown();
    ~TextMarkdown();

    void update(Config config);
    F32 naturalWidth() const;
    const TextGrid::Metrics& metrics() const;
    const std::vector<Heading>& headings() const;
    void scrollToLine(U64 line);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorOverChildren(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TEXT_MARKDOWN_HH
