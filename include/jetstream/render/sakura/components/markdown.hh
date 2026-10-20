#ifndef JETSTREAM_RENDER_SAKURA_MARKDOWN_HH
#define JETSTREAM_RENDER_SAKURA_MARKDOWN_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct Markdown : public Component {
    struct Config {
        std::string id;
        std::string value;
        F32 fontSize = Typography::FontSize;
        std::optional<F32> padding;
        bool scrollbar = false;
        Extent2D<F32> scrollbarInset = {0.0f, 0.0f};

        bool operator==(const Config&) const = default;
    };

    Markdown();
    ~Markdown();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MARKDOWN_HH
