#ifndef JETSTREAM_RENDER_SAKURA_CODE_EDITOR_HH
#define JETSTREAM_RENDER_SAKURA_CODE_EDITOR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/sakura/components/status_bar.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct CodeEditor : public Component {
    enum class Language : U8 {
        Python,
        Markdown,
    };

    using StatusTone = Sakura::StatusTone;

    struct Config {
        std::string id;
        std::string value;
        std::vector<std::string> consoleOutput;
        std::string status;
        F32 height = 200.0f;
        StatusTone statusTone = StatusTone::Info;
        bool consoleVisible = false;
        bool collapsible = false;
        F32 maxAutoHeightWindowRatio = 0.5f;
        Language language = Language::Python;
        bool lineNumbers = true;
        bool lineWrapping = false;
        bool showActiveLine = true;
        std::optional<Padding> padding;
        F32 fontSize = Typography::FontSize;
        std::string backgroundColorKey = "editor_background";
        F32 rounding = 0.0f;
        Callback<void(std::string)> onChange;
        Callback<void(std::string)> onSubmit;
        Callback<void(Extent2D<F32>)> onContentOverflow;

        bool operator==(const Config&) const = default;
    };

    CodeEditor();
    ~CodeEditor();

    void update(Config config);

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

#endif  // JETSTREAM_RENDER_SAKURA_CODE_EDITOR_HH
