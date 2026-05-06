#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_HH

#include "../components/modal_header.hh"
#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include "jetstream/instance_remote.hh"

#include <qrencode.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RemoteView : public Sakura::Component {
    struct Config {
        bool started = false;
        std::string inviteUrl;
        std::string roomId;
        std::string accessToken;
        std::vector<Instance::Remote::ClientInfo> clients;
        std::vector<std::string> waitlist;
        std::function<void()> onStart;
        std::function<void()> onConfigure;
        std::function<void()> onStop;
        std::function<void(const std::string&)> onApprove;
        std::function<void(const std::string&)> onOpenUrl;
        std::function<void(const std::string&, const std::string&)> onCopy;
    };

    void update(Config config) {
        this->config = std::move(config);
        if (this->config.started) {
            updateQrCode(this->config.inviteUrl);
        } else {
            cachedQrUrl.clear();
            cachedQrData.clear();
            cachedQrWidth = 0;
        }

        header.update({
            .id = "RemoteStreamingHeader",
            .title = ICON_FA_TOWER_BROADCAST " Remote Streaming",
            .description = "Stream your session to remote clients via WebRTC. Clients can connect by scanning the QR code or using the invite URL.",
        });

        startButton.update({
            .id = "RemoteStreamingStart",
            .str = ICON_FA_PLAY " Start Streaming",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onStart) {
                    this->config.onStart();
                }
            },
        });
        configureButton.update({
            .id = "RemoteStreamingConfigure",
            .str = ICON_FA_GEAR " Configure Remote Settings",
            .size = {-1.0f, 40.0f},
            .onClick = [this]() {
                if (this->config.onConfigure) {
                    this->config.onConfigure();
                }
            },
        });

        qrSplit.update({
            .id = "RemoteStreamingQrSplit",
            .leftWidth = 220.0f,
            .height = 260.0f,
        });
        qrCode.update({
            .id = "RemoteStreamingQrCode",
            .data = cachedQrData,
            .width = cachedQrWidth,
            .onClick = [this]() {
                openInvite();
            },
        });
        openInviteButton.update({
            .id = "RemoteStreamingOpenInvite",
            .str = ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Open in browser",
            .size = {-1.0f, 0.0f},
            .onClick = [this]() {
                openInvite();
            },
        });
        connectionTitle.update({
            .id = "RemoteStreamingConnectionTitle",
            .str = "Connection Info",
            .font = Sakura::Text::Font::H2,
            .scale = 1.05f,
        });
        connectionTable.update({
            .id = "RemoteStreamingConnectionTable",
            .columns = {"Property", "Value"},
            .rows = {
                {"Room ID", this->config.roomId},
                {"Access Token", this->config.accessToken},
            },
            .showHeaders = false,
        });
        copyRoomButton.update({
            .id = "RemoteStreamingCopyRoom",
            .str = "Copy Room ID",
            .size = {-1.0f, 0.0f},
            .onClick = [this]() {
                copy("Room ID", this->config.roomId);
            },
        });
        copyTokenButton.update({
            .id = "RemoteStreamingCopyToken",
            .str = "Copy Access Token",
            .size = {-1.0f, 0.0f},
            .onClick = [this]() {
                copy("Access token", this->config.accessToken);
            },
        });
        connectedTitle.update({
            .id = "RemoteStreamingConnectedTitle",
            .str = "Connected Clients",
            .font = Sakura::Text::Font::H2,
            .scale = 1.05f,
        });
        pendingTitle.update({
            .id = "RemoteStreamingPendingTitle",
            .str = "Pending Connections",
            .font = Sakura::Text::Font::H2,
            .scale = 1.05f,
        });
        connectedText.update({
            .id = "RemoteStreamingConnectedText",
            .str = connectedClientsText(),
            .tone = connectedClientsText().starts_with("No") ? Sakura::Text::Tone::Disabled : Sakura::Text::Tone::Success,
        });
        pendingHint.update({
            .id = "RemoteStreamingPendingHint",
            .str = "Click a pending code to approve it.",
            .tone = Sakura::Text::Tone::Disabled,
        });
        noPendingText.update({
            .id = "RemoteStreamingNoPending",
            .str = "No pending connections.",
            .tone = Sakura::Text::Tone::Disabled,
        });
        stopButton.update({
            .id = "RemoteStreamingStop",
            .str = ICON_FA_STOP " Stop Streaming",
            .size = {-1.0f, 40.0f},
            .variant = Sakura::Button::Variant::Destructive,
            .onClick = [this]() {
                if (this->config.onStop) {
                    this->config.onStop();
                }
            },
        });
        pendingButtons.resize(this->config.waitlist.size());
        for (U64 i = 0; i < pendingButtons.size(); ++i) {
            const std::string code = clientCode(this->config.waitlist[i]);
            pendingButtons[i].update({
                .id = "RemoteStreamingPending" + this->config.waitlist[i],
                .str = ICON_FA_CLOCK " " + code,
                .size = {-1.0f, 0.0f},
                .onClick = [this, code]() {
                    if (this->config.onApprove) {
                        this->config.onApprove(code);
                    }
                },
            });
        }
    }

    void render(const Sakura::Context& ctx) const {
        header.render(ctx);
        if (!config.started) {
            startButton.render(ctx);
            configureButton.render(ctx);
        } else {
            qrSplit.render(ctx, {
                [this](const Sakura::Context& ctx) {
                    qrCode.render(ctx);
                    openInviteButton.render(ctx);
                },
                [this](const Sakura::Context& ctx) {
                    connectionTitle.render(ctx);
                    connectionTable.render(ctx);
                    copyRoomButton.render(ctx);
                    copyTokenButton.render(ctx);
                    divider.render(ctx);
                    connectedTitle.render(ctx);
                    connectedText.render(ctx);
                    divider.render(ctx);
                    pendingTitle.render(ctx);
                    pendingHint.render(ctx);
                    if (pendingButtons.empty()) {
                        noPendingText.render(ctx);
                    } else {
                        for (const auto& button : pendingButtons) {
                            button.render(ctx);
                        }
                    }
                },
            });
            divider.render(ctx);
            stopButton.render(ctx);
        }
    }

 private:
    static std::string clientCode(const std::string& sessionId) {
        if (sessionId.size() <= 6) {
            std::string code = sessionId;
            std::transform(code.begin(), code.end(), code.begin(), ::toupper);
            return code;
        }
        std::string code = sessionId.substr(sessionId.size() - 6);
        std::transform(code.begin(), code.end(), code.begin(), ::toupper);
        return code;
    }

    std::string connectedClientsText() const {
        std::set<std::string> pendingIds(config.waitlist.begin(), config.waitlist.end());
        std::string text;
        for (const auto& client : config.clients) {
            if (pendingIds.contains(client.sessionId)) {
                continue;
            }
            if (!text.empty()) {
                text += "\n";
            }
            text += ICON_FA_CIRCLE_CHECK " " + clientCode(client.sessionId);
        }
        return text.empty() ? "No connected clients." : text;
    }

    void updateQrCode(const std::string& url) {
        if (cachedQrUrl == url) {
            return;
        }

        cachedQrUrl = url;
        cachedQrData.clear();
        cachedQrWidth = 0;

        QRcode* qr = QRcode_encodeString8bit(url.c_str(), 0, QR_ECLEVEL_L);
        if (!qr) {
            return;
        }

        cachedQrWidth = qr->width;
        cachedQrData.resize(cachedQrWidth * cachedQrWidth);
        for (int i = 0; i < cachedQrWidth * cachedQrWidth; i++) {
            cachedQrData[i] = qr->data[i] & 1;
        }
        QRcode_free(qr);
    }

    void openInvite() const {
        if (config.onOpenUrl) {
            config.onOpenUrl(config.inviteUrl);
        }
    }

    void copy(const std::string& label, const std::string& value) const {
        if (config.onCopy) {
            config.onCopy(label, value);
        }
    }

    Config config;
    ModalHeader header;
    Sakura::Button startButton;
    Sakura::Button configureButton;
    Sakura::SplitView qrSplit;
    Sakura::QrCode qrCode;
    Sakura::Button openInviteButton;
    Sakura::Text connectionTitle;
    Sakura::Table connectionTable;
    Sakura::Button copyRoomButton;
    Sakura::Button copyTokenButton;
    Sakura::Text connectedTitle;
    Sakura::Text connectedText;
    Sakura::Text pendingTitle;
    Sakura::Text pendingHint;
    Sakura::Text noPendingText;
    Sakura::Divider divider;
    Sakura::Button stopButton;
    std::vector<Sakura::Button> pendingButtons;
    std::string cachedQrUrl;
    std::vector<U8> cachedQrData;
    int cachedQrWidth = 0;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_HH
