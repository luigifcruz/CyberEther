#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct CodeEditor : public Component {
    struct Config {
        std::string id;
        std::string value;
        bool collapsible = false;
        Extent2D<F32> size = {0.0f, 200.0f};
        std::function<void(std::string)> onChange;
    };

    CodeEditor();
    ~CodeEditor();

    CodeEditor(CodeEditor&&) noexcept;
    CodeEditor& operator=(CodeEditor&&) noexcept;

    CodeEditor(const CodeEditor&) = delete;
    CodeEditor& operator=(const CodeEditor&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
