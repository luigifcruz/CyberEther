#ifndef JETSTREAM_RENDER_SAKURA_QR_CODE_HH
#define JETSTREAM_RENDER_SAKURA_QR_CODE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <vector>

namespace Jetstream::Sakura {

struct QrCode : public Component {
    struct Config {
        std::vector<U8> data;
        int width = 0;
        F32 heightRatio = 1.0f / 3.0f;

        bool operator==(const Config&) const = default;
    };

    QrCode();
    ~QrCode();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_QR_CODE_HH
