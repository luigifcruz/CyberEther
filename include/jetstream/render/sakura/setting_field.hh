#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct SettingField : public Component {
    using Child = std::function<void(const Context&)>;

    struct Config {
        std::string id;
        std::string label;
        std::string description;
        bool divider = true;
    };

    SettingField();
    ~SettingField();

    SettingField(SettingField&&) noexcept;
    SettingField& operator=(SettingField&&) noexcept;

    SettingField(const SettingField&) = delete;
    SettingField& operator=(const SettingField&) = delete;

    bool update(Config config);
    void render(const Context& ctx, Child child) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
