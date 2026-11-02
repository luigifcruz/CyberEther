#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_QR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_QR_HH

#include "../components/modal_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <qrencode.h>

#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RemoteQrView {
    static constexpr F32 HeightRatio = 1.0f / 3.0f;

    struct Config {
        std::string inviteUrl;
    };

    void update(Config config) {
        encode(config.inviteUrl);
        header.update({
            .title = ICON_LC_QR_CODE " Scan to Connect",
            .description = "Scan with another device to watch this session.",
        });
        code.update({.data = data, .width = width, .heightRatio = HeightRatio});
        spacing.update({});
        stack.update({
            .items = {
                {.child = &header},
                {.child = &code},
                {.child = &spacing},
            },
        });
        panel.update({
            .id = "RemoteQr",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

 private:
    void encode(const std::string& next) {
        if (next == url) {
            return;
        }
        url = next;
        data.clear();
        width = 0;
        if (url.empty()) {
            return;
        }
        QRcode* qr = QRcode_encodeString8bit(url.c_str(), 0, QR_ECLEVEL_L);
        if (!qr) {
            return;
        }
        width = qr->width;
        data.resize(static_cast<U64>(width) * static_cast<U64>(width));
        for (U64 index = 0; index < data.size(); ++index) {
            data[index] = qr->data[index] & 1;
        }
        QRcode_free(qr);
    }

    ModalHeader header;
    Sakura::QrCode code;
    Sakura::Spacer spacing;
    Sakura::Stack stack;
    Sakura::Panel panel;
    std::string url;
    std::vector<U8> data;
    int width = 0;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_QR_HH
