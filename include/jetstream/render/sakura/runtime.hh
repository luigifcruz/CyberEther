#pragma once

#include <jetstream/render/sakura/color.hh>
#include <jetstream/render/sakura/context.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Render {

class Window;

}  // namespace Jetstream::Render

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
    };

    struct Config {
        const Palette* palette = nullptr;
        const Render::Window* render = nullptr;
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
    void syncNodeContexts(const std::vector<std::string>& ids);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
