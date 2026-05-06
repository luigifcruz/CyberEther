#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct HStack : public Component {
    using Child = std::function<void(const Context&)>;
    using Children = std::vector<Child>;

    struct Config {
        std::string id;
        F32 spacing = 0.0f;
    };

    HStack();
    ~HStack();

    HStack(HStack&&) noexcept;
    HStack& operator=(HStack&&) noexcept;

    HStack(const HStack&) = delete;
    HStack& operator=(const HStack&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Children children) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
