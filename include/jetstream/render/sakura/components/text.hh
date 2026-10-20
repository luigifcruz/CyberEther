#ifndef JETSTREAM_RENDER_SAKURA_TEXT_HH
#define JETSTREAM_RENDER_SAKURA_TEXT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Text : public Component {
    enum class Font {
        Body,
        H1,
        H2,
        Bold,
        Display,
        DisplaySemiBold,
    };

    enum class Tone {
        Primary,
        Secondary,
        Disabled,
        Accent,
        Success,
        Warning,
    };

    enum class Align {
        Left,
        Center,
        Right,
    };

    struct Config {
        std::string str;
        Font font = Font::Body;
        Tone tone = Tone::Primary;
        Align align = Align::Left;
        std::string colorKey;
        bool wrapped = false;
        F32 scale = 1.0f;
        F32 verticalOffset = 0.0f;
        U64 maxCharacters = 256;
        F32 opacity = 1.0f;

        bool operator==(const Config&) const = default;
    };

    Text();
    ~Text();

    void update(Config config);
    const Config& config() const;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TEXT_HH
