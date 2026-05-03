#pragma once

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/context.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct QrCode : public Component {
    struct Config {
        std::string id;
        std::vector<U8> data;
        int width = 0;
        F32 moduleSize = 4.0f;
        std::function<void()> onClick;
    };

    QrCode();
    ~QrCode();

    QrCode(QrCode&&) noexcept;
    QrCode& operator=(QrCode&&) noexcept;

    QrCode(const QrCode&) = delete;
    QrCode& operator=(const QrCode&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura
