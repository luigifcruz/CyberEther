#ifndef JETSTREAM_RENDER_SAKURA_MARKDOWN_HH
#define JETSTREAM_RENDER_SAKURA_MARKDOWN_HH

#include <jetstream/render/sakura/callback.hh>
#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/text_markdown.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Markdown : public Component {
    struct Config {
        std::string id;
        std::string value;
        F32 fontSize = Typography::FontSize;
        std::optional<Padding> padding;
        bool scrollbar = false;
        Extent2D<F32> scrollbarInset = {0.0f, 0.0f};
        Callback<bool(const std::string&)> onLink;
        std::optional<TextMarkdown::Hero> hero;
        bool softBreaks = false;
        F32 maxWidth = 0.0f;

        bool operator==(const Config&) const = default;
    };

    Markdown();
    ~Markdown();

    void update(Config config);
    const std::vector<TextMarkdown::Heading>& headings() const;
    const TextGrid::Metrics& metrics() const;
    void scrollToLine(U64 line);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MARKDOWN_HH
