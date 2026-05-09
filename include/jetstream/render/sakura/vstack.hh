#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct VStack : public Component {
    using Child = std::function<void(const Context&)>;
    using Children = std::vector<Child>;

    struct Config {
        std::string id;
        F32 spacing = 0.0f;
    };

    VStack();
    ~VStack();

    VStack(VStack&&) noexcept;
    VStack& operator=(VStack&&) noexcept;

    VStack(const VStack&) = delete;
    VStack& operator=(const VStack&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Children children) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
