#ifndef JETSTREAM_RENDER_SAKURA_HIGHLIGHT_TEXT_HH
#define JETSTREAM_RENDER_SAKURA_HIGHLIGHT_TEXT_HH

#include <jetstream/render/sakura/callback.hh>
#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct HighlightText : public Component {
    struct Mark {
        U64 begin = 0;
        U64 end = 0;

        bool operator==(const Mark&) const = default;
    };

    struct Config {
        std::string id;
        std::string str;
        std::vector<Mark> marks;
        F32 fontSize = Typography::FontSize;
        std::string fontName = Typography::BodyFont;
        std::string colorKey = "text_secondary";
        std::string markColorKey = "text_primary";
        std::string markBackgroundColorKey = "editor_selection_match";
        Callback<void()> onClick;

        bool operator==(const Config&) const = default;
    };

    HighlightText();
    ~HighlightText();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_HIGHLIGHT_TEXT_HH
