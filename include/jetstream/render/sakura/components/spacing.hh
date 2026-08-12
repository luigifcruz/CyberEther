#ifndef JETSTREAM_RENDER_SAKURA_SPACING_HH
#define JETSTREAM_RENDER_SAKURA_SPACING_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Spacing {
    struct Config {
        std::string id;
        U64 lines = 1;
    };

    Spacing();
    ~Spacing();

    Spacing(Spacing&&) noexcept;
    Spacing& operator=(Spacing&&) noexcept;

    Spacing(const Spacing&) = delete;
    Spacing& operator=(const Spacing&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SPACING_HH
