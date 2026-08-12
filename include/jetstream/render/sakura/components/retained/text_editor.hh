#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_EDITOR_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_EDITOR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/text_grid.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct TextEditor : public Component {
    enum class Language : U8 {
        Python,
        Markdown,
    };

    struct Config {
        std::string id;
        std::string value;
        F32 fontSize = 15.0f;
        std::string fontName = "default_mono";
        bool monospace = true;
        bool lineNumbers = true;
        TextGrid::Wrap wrap = TextGrid::Wrap::None;
        Language language = Language::Python;
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
        std::vector<std::string> styleColorKeys = {"editor_comment", "editor_keyword", "editor_string", "editor_constant",
                                                   "editor_function", "editor_type", "editor_constant",
                                                   "editor_keyword", "editor_type"};
        std::vector<std::string> styleFonts;
        std::vector<std::string> styleBackgroundColorKeys;
        std::function<void(std::string)> onChange;
        std::function<void(std::string)> onSubmit;
    };

    TextEditor();
    ~TextEditor();

    bool update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_EDITOR_HH
