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
            .description = "Share this session with remote clients via WebRTC.",
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
            .leftWidth = 190.0f,
            .height = 300.0f,
        });
        qrCode.update({
            .id = "RemoteStreamingQrCode",
            .data = cachedQrData,
            .width = cachedQrWidth,
            .moduleSize = 5.0f,
            .onClick = [this]() {
                openInvite();
            },
        });
        openInviteButton.update({
            .id = "RemoteStreamingOpenInvite",
            .str = ICON_FA_ARROW_UP_RIGHT_FROM_SQUARE " Open in browser",
            .size = {-1.0f, 34.0f},
            .onClick = [this]() {
                openInvite();
            },
        });
        copyTokenButton.update({
            .id = "RemoteStreamingCopyToken",
            .str = "Copy Access Token",
            .size = {-1.0f, 34.0f},
            .onClick = [this]() {
                copy("Access token", this->config.accessToken);
            },
        });
        clientTable.update({
            .id = "RemoteStreamingClientsTable",
            .columns = {"Client", "Status"},
            .fixedColumnWidths = {0.0f, 160.0f},
            .size = {0.0f, 270.0f},
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
        clientRows = buildClientRows();
        clientCodeTexts.resize(clientRows.size());
        clientApproveButtons.resize(clientRows.size());
        for (U64 i = 0; i < clientRows.size(); ++i) {
            const auto& row = clientRows[i];
            clientCodeTexts[i].update({
                .id = "RemoteStreamingClient" + row.sessionId,
                .str = row.code,
            });
            clientApproveButtons[i].update({
                .id = "RemoteStreamingApprove" + row.sessionId,
                .str = row.pending ? "Approve" : "Approved",
                .size = {-1.0f, 0.0f},
                .disabled = !row.pending,
                .onClick = [this, code = row.code]() {
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
                    copyTokenButton.render(ctx);
                },
                [this](const Sakura::Context& ctx) {
                    renderClientTable(ctx);
                },
            });
            divider.render(ctx);
            stopButton.render(ctx);
        }
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

    void renderClientTable(const Sakura::Context& ctx) const {
        Sakura::Table::Rows rows;
        rows.reserve(clientRows.size());
        for (U64 i = 0; i < clientRows.size(); ++i) {
            Sakura::Table::Row row;
            row.push_back([this, i](const Sakura::Context& ctx) { clientCodeTexts[i].render(ctx); });
            row.push_back([this, i](const Sakura::Context& ctx) {
                clientApproveButtons[i].render(ctx);
            });
            rows.push_back(std::move(row));
        }
        clientTable.render(ctx, std::move(rows));
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
    Sakura::Button copyTokenButton;
    Sakura::Table clientTable;
    Sakura::Divider divider;
    Sakura::Button stopButton;
    std::vector<ClientRow> clientRows;
    std::vector<Sakura::Text> clientCodeTexts;
    std::vector<Sakura::Button> clientApproveButtons;
    std::string cachedQrUrl;
    std::vector<U8> cachedQrData;
    int cachedQrWidth = 0;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_REMOTE_HH
