#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_HH

#include "../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include "jetstream/instance_remote.hh"

#include <qrencode.h>

#include <algorithm>
#include <cctype>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct RemoteView {
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
        using Button = Sakura::Button;

        this->config = std::move(config);
        if (this->config.started) {
            updateQrCode(this->config.inviteUrl);
        } else {
            cachedQrUrl.clear();
            cachedQrData.clear();
            cachedQrWidth = 0;
        }
        header.update({
            .title = ICON_FA_TOWER_BROADCAST " Remote Streaming",
            .description = "Share this session with remote clients via WebRTC.",
        });
        startButton.update({
            .str = ICON_FA_PLAY " Start Streaming",
            .size = {Sakura::Fill, 40.0f},
            .variant = Button::Variant::Action,
            .onClick = [this]() {
                if (this->config.onStart) {
                    this->config.onStart();
                }
            },
        });
        configureButton.update({
            .str = ICON_FA_GEAR " Configure Remote Settings",
            .size = {Sakura::Fill, 40.0f},
            .onClick = [this]() {
                if (this->config.onConfigure) {
                    this->config.onConfigure();
                }
            },
        });
        qrCode.update({
            .data = cachedQrData,
            .width = cachedQrWidth,
            .moduleSize = 5.0f,
            .onClick = [this]() {
                openInvite();
            },
        });
        openInviteButton.update({
            .str = ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Open in browser",
            .size = {Sakura::Fill, 34.0f},
            .onClick = [this]() {
                openInvite();
            },
        });
        copyTokenButton.update({
            .str = "Copy Access Token",
            .size = {Sakura::Fill, 34.0f},
            .onClick = [this]() {
                copy("Access token", this->config.accessToken);
            },
        });
        stopButton.update({
            .str = ICON_FA_STOP " Stop Streaming",
            .size = {Sakura::Fill, 40.0f},
            .variant = Button::Variant::Destructive,
            .onClick = [this]() {
                if (this->config.onStop) {
                    this->config.onStop();
                }
            },
        });
        divider.update({});

        clientRows = buildClientRows();
        while (clientCells.size() < clientRows.size()) {
            clientCells.push_back(std::make_unique<ClientCells>());
        }
        std::vector<std::vector<Sakura::Component*>> cellRows;
        for (U64 i = 0; i < clientRows.size(); ++i) {
            const auto& row = clientRows[i];
            auto& cells = *clientCells[i];
            cells.code.update({.str = row.code});
            cells.approve.update({
                .str = row.pending ? "Approve" : "Approved",
                .size = {Sakura::Fill, 0.0f},
                .disabled = !row.pending,
                .onClick = [this, code = row.code]() {
                    if (this->config.onApprove) {
                        this->config.onApprove(code);
                    }
                },
            });
            cellRows.push_back({&cells.code, &cells.approve});
        }
        clientTable.update({
            .columns = {"Client", "Status"},
            .cellRows = std::move(cellRows),
            .fixedColumnWidths = {0.0f, 160.0f},
            .size = {Sakura::Fill, 270.0f},
        });

        qrColumn.update({
            .items = {
                {.child = &qrCode},
                {.child = &openInviteButton},
                {.child = &copyTokenButton},
            },
        });
        qrSplit.update({
            .left = &qrColumn,
            .right = &clientTable,
            .leftWidth = 190.0f,
            .height = 300.0f,
        });

        std::vector<Sakura::Stack::Item> items = {{.child = &header}};
        if (!this->config.started) {
            items.push_back({.child = &startButton});
            items.push_back({.child = &configureButton});
        } else {
            items.push_back({.child = &qrSplit});
            items.push_back({.child = &divider});
            items.push_back({.child = &stopButton});
        }
        stack.update({.items = std::move(items)});
        panel.update({
            .id = "RemoteStreaming",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::Component& component() {
        return panel;
    }

 private:
    struct ClientRow {
        std::string sessionId;
        std::string code;
        bool pending = false;
    };

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

    std::vector<ClientRow> buildClientRows() const {
        std::set<std::string> pendingIds(config.waitlist.begin(), config.waitlist.end());
        std::set<std::string> seenIds;
        std::vector<ClientRow> rows;
        for (const auto& client : config.clients) {
            if (client.sessionId.empty()) {
                continue;
            }
            seenIds.insert(client.sessionId);
            rows.push_back({
                .sessionId = client.sessionId,
                .code = clientCode(client.sessionId),
                .pending = pendingIds.contains(client.sessionId),
            });
        }
        for (const auto& sessionId : config.waitlist) {
            if (sessionId.empty() || seenIds.contains(sessionId)) {
                continue;
            }
            rows.push_back({
                .sessionId = sessionId,
                .code = clientCode(sessionId),
                .pending = true,
            });
        }
        return rows;
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

    struct ClientCells {
        Sakura::Text code;
        Sakura::Button approve;
    };

    Config config;
    ModalHeader header;
    Sakura::Button startButton;
    Sakura::Button configureButton;
    Sakura::QrCode qrCode;
    Sakura::Button openInviteButton;
    Sakura::Button copyTokenButton;
    Sakura::Stack qrColumn;
    std::vector<ClientRow> clientRows;
    std::vector<std::unique_ptr<ClientCells>> clientCells;
    Sakura::Table clientTable;
    Sakura::SplitView qrSplit;
    Sakura::Divider divider;
    Sakura::Button stopButton;
    Sakura::Stack stack;
    Sakura::Panel panel;
    std::string cachedQrUrl;
    std::vector<U8> cachedQrData;
    int cachedQrWidth = 0;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_HH
