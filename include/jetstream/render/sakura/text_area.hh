#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct TextArea : public Component {
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
