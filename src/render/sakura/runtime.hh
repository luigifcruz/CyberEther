#ifndef JETSTREAM_RENDER_SAKURA_RUNTIME_HH
#define JETSTREAM_RENDER_SAKURA_RUNTIME_HH

#include "context.hh"

#include <memory>

namespace Jetstream::Sakura {

struct Runtime {
    struct FontData {
        const unsigned int* data = nullptr;
        unsigned int size = 0;

        bool valid() const;
    };

    struct FontConfig {
        FontData body;
        FontData bold;
        FontData iconRegular;
        FontData iconSolid;
        FontData display;
    };

    struct Config {
        const Palette* palette = nullptr;
        Render::Window* render = nullptr;
    };

    Runtime();
    ~Runtime();

    Runtime(Runtime&&) noexcept;
    Runtime& operator=(Runtime&&) noexcept;

    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void create(FontConfig fontConfig);
    void update(Config config);
    Context context();

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_RUNTIME_HH
