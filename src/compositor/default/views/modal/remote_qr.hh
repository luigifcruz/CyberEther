#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_QR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_QR_HH

#include "../components/card_button.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <qrencode.h>

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RemoteQrView {
    static constexpr F32 HeightRatio = 1.0f / 3.0f;
    static constexpr F32 ButtonHeight = 36.0f;
    static constexpr F32 TileRounding = 16.0f;

    struct Config {
        std::string inviteUrl;
        Sakura::Callback<void()> onCopyLink;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    void encode(const std::string& next);

    Config config;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Stack titleRow;
    Sakura::Text description;
    Sakura::Stack header;
    Sakura::QrCode code;
    CardButton close;
    CardButton copy;
    Sakura::Grid footer;
    Sakura::Stack stack;
    Sakura::Panel panel;
    std::string url;
    std::vector<U8> data;
    int width = 0;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_QR_HH
