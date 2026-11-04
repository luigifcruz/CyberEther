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

    void update(Config config) {
        this->config = std::move(config);
        encode(this->config.inviteUrl);

        using Text = Sakura::Text;

        icon.update({
            .str = ICON_LC_QR_CODE,
            .colorKey = "text_primary",
            .scale = Sakura::Typography::Scale::Heading,
        });
        title.update({
            .str = "Scan to Connect",
            .font = Text::Font::Display,
            .scale = Sakura::Typography::Scale::Headline,
        });
        titleRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &icon},
                {.child = &title},
            },
        });
        description.update({
            .str = "Point another device's camera at the code to watch this session.",
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .wrapped = true,
            .scale = Sakura::Typography::Scale::Lead,
        });
        header.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = {
                {.child = &titleRow},
                {.child = &description},
            },
        });

        code.update({.data = data, .width = width, .heightRatio = HeightRatio, .rounding = TileRounding});

        close.update({
            .str = "Close",
            .height = ButtonHeight,
            .onClick = [this]() {
                if (this->config.onClose) {
                    this->config.onClose();
                }
            },
        });
        copy.update({
            .str = ICON_LC_COPY "  Copy Link",
            .tone = CardButton::Tone::Action,
            .height = ButtonHeight,
            .onClick = [this]() {
                if (this->config.onCopyLink) {
                    this->config.onCopyLink();
                }
            },
        });
        footer.update({
            .columns = 2,
            .cellPadding = {3.0f, 0.0f},
            .outerPadding = false,
            .items = {&close, &copy},
        });

        stack.update({
            .spacing = Sakura::Spacing::Large,
            .items = {
                {.child = &header},
                {.child = &code},
                {.child = &footer},
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
