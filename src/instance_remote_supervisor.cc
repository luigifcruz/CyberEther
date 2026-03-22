#include "jetstream/detail/instance_remote_supervisor.hh"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <iostream>
#include <qrencode.h>

namespace Jetstream {

Instance::Remote::Supervisor::Supervisor(Instance::Remote* remote, bool autoJoin_)
    : remote_(remote), autoJoin(autoJoin_) {}

Instance::Remote::Supervisor::~Supervisor() {
    stop();
}

void Instance::Remote::Supervisor::start() {
    if (running_ || !remote_) {
        return;
    }

    running_ = true;
    worker_ = std::thread(&Supervisor::run, this);
}

void Instance::Remote::Supervisor::stop() {
    running_ = false;
    if (worker_.joinable()) {
        worker_.join();
    }
}

void Instance::Remote::Supervisor::run() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::seconds(2));

        if (!running_) {
            break;
        }

        const auto& waitlist = remote_->waitlist();
        if (waitlist.empty()) {
            continue;
        }

        for (const auto& sessionId : waitlist) {
            if (seenSessions.count(sessionId)) {
                continue;
            }
            seenSessions.insert(sessionId);

            if (sessionId.size() < 6) {
                continue;
            }

            std::string code = sessionId.substr(sessionId.length() - 6);
            std::transform(code.begin(), code.end(), code.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });

            if (autoJoin) {
                remote_->approveClient(code);
                continue;
            }

            if (prompt(code)) {
                remote_->approveClient(code);
            }
        }
    }
}

void Instance::Remote::Supervisor::print() const {
    QRcode* qr = QRcode_encodeString8bit(remote_->inviteUrl().c_str(), 0, QR_ECLEVEL_L);
    if (!qr) {
        jst::fmt::print(stderr, "[QR encode error]\n");
        return;
    }

    const int qrWidth = qr->width;
    const int border = 2;
    const int totalWidth = qrWidth + border * 2;
    const int boxInner = totalWidth + 4;

    auto isBlack = [&](int x, int y) -> bool {
        if (x < 0 || y < 0 || x >= qrWidth || y >= qrWidth) return false;
        return (qr->data[y * qrWidth + x] & 1) != 0;
    };

    std::string hLine;
    for (int i = 0; i < boxInner; ++i) hLine += "═";

    auto printCentered = [&](const std::string& text) {
        int totalPad = boxInner - static_cast<int>(text.length());
        int left = totalPad / 2;
        int right = totalPad - left;
        jst::fmt::print("║{:>{}}{}{:>{}}\n", "", left, text, "║", right + 1);
    };

    jst::fmt::print("\n╔{}╗\n", hLine);
    printCentered("CyberEther Remote");
    printCentered("Scan QR code or open link to connect");
    jst::fmt::print("╠{}╣\n", hLine);

    for (int y = -border; y < qrWidth + border; y += 2) {
        std::string row;
        for (int x = -border; x < qrWidth + border; ++x) {
            const bool upper = isBlack(x, y);
            const bool lower = isBlack(x, y + 1);
            if (upper && lower)      row += "█";
            else if (upper)          row += "▀";
            else if (lower)          row += "▄";
            else                     row += " ";
        }
        jst::fmt::print("║  {}  ║\n", row);
    }

    jst::fmt::print("╚{}╝\n\n", hLine);

    QRcode_free(qr);

    jst::fmt::print("Room ID:      {}\n", remote_->roomId());
    jst::fmt::print("Join URL:     {}\n", remote_->inviteUrl());
    jst::fmt::print("Access Token: {}\n\n", remote_->accessToken());
}

bool Instance::Remote::Supervisor::prompt(const std::string& code) const {
    const int boxInner = 38;

    std::string hLine;
    for (int i = 0; i < boxInner; ++i) hLine += "═";

    auto printCentered = [&](const std::string& text) {
        int totalPad = boxInner - static_cast<int>(text.length());
        int left = totalPad / 2;
        int right = totalPad - left;
        jst::fmt::print("║{:>{}}{}{:>{}}\n", "", left, text, "║", right + 1);
    };

    jst::fmt::print("\n╔{}╗\n", hLine);
    printCentered("New Connection Request");
    printCentered("Verify client code before approving");
    jst::fmt::print("╠{}╣\n", hLine);
    printCentered(code);
    jst::fmt::print("╚{}╝\n\n", hLine);

    jst::fmt::print("Approve? [Y/n]: ");
    std::fflush(stdout);

    std::string input;
    if (std::getline(std::cin, input)) {
        return input.empty() || input == "y" || input == "Y";
    }

    return false;
}

}  // namespace Jetstream
