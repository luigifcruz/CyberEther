#ifndef JETSTREAM_RENDER_SAKURA_SETTING_FIELD_HH
#define JETSTREAM_RENDER_SAKURA_SETTING_FIELD_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct SettingField : public Component {
    struct Config {
        std::string label;
        std::string description;
        Component* control = nullptr;
        bool divider = true;

        bool operator==(const Config&) const = default;
    };

    SettingField();
    ~SettingField();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SETTING_FIELD_HH
