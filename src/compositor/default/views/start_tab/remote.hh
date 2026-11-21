#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_REMOTE_HH

#include "../components/card_button.hh"
#include "../components/section_group.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct RemoteSectionView {
    static constexpr F32 ButtonHeight = 36.0f;
    static constexpr F32 RowButtonSize = 32.0f;

    struct Viewer {
        std::string id;
        std::string code;
        bool waiting = false;
    };

    struct Config {
        bool started = false;
        std::vector<Viewer> viewers;
        Sakura::Callback<void()> onStart;
        Sakura::Callback<void()> onStop;
        Sakura::Callback<void()> onSettings;
        Sakura::Callback<void()> onCopyLink;
        Sakura::Callback<void()> onShowQr;
        Sakura::Callback<void(const std::string&)> onApprove;
        Sakura::Callback<void(const std::string&)> onRemove;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    struct ViewerRow {
        std::string id;
        bool listed = false;
        Sakura::Text icon;
        Sakura::Text code;
        Sakura::Text status;
        Sakura::Stack texts;
        CardButton approve;
        CardButton remove;
        Sakura::Stack actions;
        Sakura::Stack row;
        Sakura::Div card;
    };

    static void tinted(Sakura::Button& button, const char* label, const std::string& tone,
                       Sakura::Callback<void()> onClick);
    ViewerRow& rowFor(const std::string& id, const std::unordered_set<std::string>& active);
    std::vector<Sakura::Component*> updateViewers();

    Config config;
    CardButton startButton;
    Sakura::Button stopButton;
    CardButton settingsButton;
    CardButton qrButton;
    CardButton copyButton;
    Sakura::Stack primaryRow;
    Sakura::Grid shareRow;
    std::vector<std::unique_ptr<ViewerRow>> viewerRows;
    Sakura::Text emptyText;
    Sakura::Div emptyCard;
    Sakura::Stack body;
    SectionGroup group;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_REMOTE_HH
