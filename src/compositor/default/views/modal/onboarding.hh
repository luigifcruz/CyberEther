#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ONBOARDING_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ONBOARDING_HH

#include "../components/card_button.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct OnboardingView {
    static constexpr F32 Width = 600.0f;
    static constexpr F32 HeroHeight = 280.0f;
    static constexpr F32 HeroRounding = 16.0f;
    static constexpr F32 WordmarkScale = 3.2f;
    static constexpr F32 WordmarkEmboss = 0.0125f;
    static constexpr F32 TileRounding = 16.0f;
    static constexpr F32 TileIconScale = 1.9f;
    static constexpr F32 ButtonHeight = 36.0f;
    static constexpr F32 NoteFontSize = 17.0f;
    static constexpr F32 RuleOpacity = 0.18f;

    struct Feature {
        std::string icon;
        std::string title;
        std::string text;
        std::string colorKey = "cyber_blue";
    };

    struct Section {
        std::string title;
        std::string colorKey = "text_primary";
        std::vector<Feature> features;
    };

    struct Config {
        std::string icon;
        std::string title;
        std::string subtitle;
        std::string action;
        std::string version;
        std::vector<Feature> features;
        std::vector<Section> sections;
        Sakura::Callback<void()> onDismiss;
    };

    static Extent2D<F32> Size();
    static std::string MajorVersion(const std::string& version);

    struct Artwork : public Sakura::Div {
        void update(const std::string& version, const std::vector<Feature>& features);

     private:
        struct Tile {
            Sakura::Text icon;
            Sakura::Div box;
        };

        Sakura::Text wordmark;
        Sakura::Text release;
        Sakura::Spacer markLeft;
        Sakura::Spacer markRight;
        Sakura::Stack mark;
        std::vector<std::unique_ptr<Tile>> tiles;
        Sakura::Spacer tilesLeft;
        Sakura::Spacer tilesRight;
        Sakura::Stack tileRow;
        Sakura::Spacer top;
        Sakura::Spacer bottom;
        Sakura::Stack stack;
    };

    struct NoteList : public Sakura::Stack {
        void update(const std::vector<Section>& sections);

     private:
        struct Row {
            Sakura::Text icon;
            Sakura::Markdown body;
            Sakura::Stack stack;
        };

        struct Group {
            Sakura::Text label;
            Sakura::WaveLine rule;
            Sakura::Stack heading;
            std::vector<std::unique_ptr<Row>> rows;
            Sakura::Stack notes;
            Sakura::Stack stack;
        };

        std::vector<std::unique_ptr<Group>> groups;
    };

    struct Page {
        void update(const Config& config);
        Sakura::Component& component();

     private:
        const Config* config = nullptr;
        Sakura::Text icon;
        Sakura::Text title;
        Sakura::Stack titleRow;
        Sakura::Text subtitle;
        Sakura::Stack header;
        Artwork hero;
        NoteList highlights;
        CardButton proceed;
        Sakura::Stack stack;
        Sakura::Panel panel;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    Page page;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ONBOARDING_HH
