#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Text : public Component {
    enum class Align {
        Left,
        Center,
        Right,
    };

    enum class Font {
        Current,
        Body,
        H1,
        H2,
        Bold,
    };

    enum class Tone {
        Primary,
        Secondary,
        Disabled,
        Accent,
        Success,
        Warning,
    };

    struct Config {
        std::string id;
        std::string str;
        Font font = Font::Current;
        Tone tone = Tone::Primary;
        Align align = Align::Left;
        std::string colorKey;
        bool wrapped = false;
        F32 scale = 1.0f;
    };

    Text();
    ~Text();

    Text(Text&&) noexcept;
    Text& operator=(Text&&) noexcept;

    Text(const Text&) = delete;
    Text& operator=(const Text&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
