#ifndef JETSTREAM_RENDER_SAKURA_MODAL_HH
#define JETSTREAM_RENDER_SAKURA_MODAL_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct Modal {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        std::optional<Extent2D<F32>> size;
        F32 minWidth = 550.0f;
        std::function<void()> onOpen;
        std::function<void()> onClose;
    };

    Modal();
    ~Modal();

    Modal(Modal&&) noexcept;
    Modal& operator=(Modal&&) noexcept;

    Modal(const Modal&) = delete;
    Modal& operator=(const Modal&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MODAL_HH
