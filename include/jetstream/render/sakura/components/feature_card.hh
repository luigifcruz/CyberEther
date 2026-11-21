#ifndef JETSTREAM_RENDER_SAKURA_FEATURE_CARD_HH
#define JETSTREAM_RENDER_SAKURA_FEATURE_CARD_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct FeatureCard : public Component {
    struct Config {
        std::string title;
        std::string description;
        Component* action = nullptr;
        Component* content = nullptr;

        bool operator==(const Config&) const = default;
    };

    FeatureCard();
    ~FeatureCard();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_FEATURE_CARD_HH
