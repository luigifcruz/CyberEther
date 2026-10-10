#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_COMPOSER_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_COMPOSER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/model_picker.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct ChatComposer : public Component {
    struct Style {
        std::string placeholder;
        std::string icon;
        std::string iconColorKey = "accent_color";
        std::string backgroundColorKey = "card";
        std::string borderColorKey = "border";
        F32 fontSize = Typography::FontSize;
        F32 borderWidth = 1.0f;
        std::optional<F32> cornerRadius;
        bool compact = false;
        F32 minLines = 1.0f;
        F32 maxLines = 6.0f;
        F32 padding = 7.0f;
        Padding margin;
        bool clearButton = true;

        bool operator==(const Style&) const = default;
    };

    struct Selector {
        std::vector<ModelPicker::Option> options;
        std::string value;
        std::string placeholder;
        bool disabled = false;
        std::vector<ModelPicker::Option> efforts;
        std::string effort;
        std::function<void(const std::string&)> onSelect;
        std::function<void(const std::string&)> onEffort;
    };

    struct Usage {
        F32 value = 0.0f;
        std::string title;
        std::vector<std::string> details;

        bool operator==(const Usage&) const = default;
    };

    struct Config {
        std::string id;
        Style style;
        bool busy = false;
        std::optional<Selector> selector;
        std::optional<Usage> usage;
        std::function<void(const std::string&)> onSubmit;
        std::function<void()> onCancel;
        std::function<void()> onClear;
        std::function<void()> onAttach;
        std::function<void()> onVoice;
        U64 focusRequest = 0;
        U64 clearRequest = 0;
    };

    ChatComposer();
    ~ChatComposer();

    ChatComposer(const ChatComposer&) = delete;
    ChatComposer& operator=(const ChatComposer&) = delete;

    bool update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool event(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_COMPOSER_HH
