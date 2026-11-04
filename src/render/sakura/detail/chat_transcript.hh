#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_CHAT_TRANSCRIPT_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_CHAT_TRANSCRIPT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/chat.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct ChatTranscript : public Component {
    static constexpr U64 UserSlots = 8;
    static constexpr U64 MarkdownSlots = 12;
    static constexpr U64 DisclosureSlots = 12;
    static constexpr U64 ShimmerSteps = 48;

    struct Config {
        std::string id;
        Chat::Snapshot snapshot;
        F32 fontSize = Typography::FontSize;
        std::string surfaceColorKey = "background";
        U64 tailRequest = 0;

        bool operator==(const Config&) const = default;
    };

    ChatTranscript();
    ~ChatTranscript();

    void update(Config config);

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_CHAT_TRANSCRIPT_HH
