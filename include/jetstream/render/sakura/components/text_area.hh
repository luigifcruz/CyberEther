#ifndef JETSTREAM_RENDER_SAKURA_TEXT_AREA_HH
#define JETSTREAM_RENDER_SAKURA_TEXT_AREA_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct TextArea {
    enum class Submit {
        OnEdit,
        OnEnter,
        OnCommit,
    };

    struct Config {
        std::string id;
        std::string value;
        Extent2D<F32> size = {0.0f, 80.0f};
        Submit submit = Submit::OnEdit;
        std::function<void(const std::string&)> onChange;
    };

    TextArea();
    ~TextArea();

    TextArea(TextArea&&) noexcept;
    TextArea& operator=(TextArea&&) noexcept;

    TextArea(const TextArea&) = delete;
    TextArea& operator=(const TextArea&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TEXT_AREA_HH
