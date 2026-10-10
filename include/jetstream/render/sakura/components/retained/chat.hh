#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/chat_composer.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct Chat : public Component {
    enum class Role { User, Assistant };
    enum class PartKind { Markdown, Disclosure };

    struct Part {
        PartKind kind = PartKind::Markdown;
        U64 key = 0;
        U64 revision = 0;
        std::string text;
        std::string label;
        std::string summary;
        bool busy = false;
        bool expandable = true;

        bool operator==(const Part&) const = default;
    };

    struct Message {
        Role role;
        std::string text;
        std::vector<Part> parts;

        bool operator==(const Message&) const = default;
    };

    struct Snapshot {
        struct Messages {
            using History = std::vector<std::shared_ptr<const Message>>;
            std::shared_ptr<const History> history = std::make_shared<const History>();
            std::shared_ptr<const Message> tail;

            U64 size() const { return history->size() + (tail ? 1 : 0); }
            bool empty() const { return size() == 0; }
            const Message& operator[](U64 index) const { return *at(index); }
            const std::shared_ptr<const Message>& at(U64 index) const {
                return index < history->size() ? (*history)[index] : tail;
            }
            void push_back(Message message) {
                if (tail) {
                    auto next = std::make_shared<History>(*history);
                    next->push_back(tail);
                    history = std::move(next);
                }
                tail = std::make_shared<const Message>(std::move(message));
            }
            bool operator==(const Messages&) const = default;
        };
        U64 generation = 0;
        bool streaming = false;
        Messages messages;

        bool operator==(const Snapshot&) const = default;
    };

    using ComposerStyle = ChatComposer::Style;

    struct Config {
        std::string id;
        Snapshot snapshot;
        std::optional<ChatComposer::Selector> selector;
        std::optional<ChatComposer::Usage> usage;
        F32 fontSize = Typography::FontSize;
        F32 maxContentWidth = 0.0f;
        F32 transcriptOpacity = 1.0f;
        std::string backgroundColorKey = "background";
        std::string surfaceColorKey = "background";
        ComposerStyle composer{.minLines = 2.0f};
        std::function<void(const std::string&)> onSubmit;
        std::function<void()> onCancel;
        std::function<void()> onClear;
        std::function<void()> onVoice;
        U64 focusRequest = 0;
        U64 clearRequest = 0;
    };

    Chat();
    ~Chat();

    Chat(const Chat&) = delete;
    Chat& operator=(const Chat&) = delete;

    bool update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_HH
