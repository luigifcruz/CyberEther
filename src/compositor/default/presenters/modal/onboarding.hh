#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_ONBOARDING_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_ONBOARDING_HH

#include "../context.hh"

#include "../../model/messages.hh"
#include "../../views/modal/onboarding.hh"

#include "jetstream/config.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <vector>

namespace Jetstream {

struct OnboardingModalPresenter {
    const PresenterContext& context;

    explicit OnboardingModalPresenter(const PresenterContext& context) : context(context) {}

    static constexpr const char* Release = "2.0";

    OnboardingView::Config build() const {
        const auto enqueue = context.callbacks.enqueueMail;

        return OnboardingView::Config{
            .version = Release,
            .features = Highlights(),
            .sections = Sections(),
            .onDismiss = [enqueue]() {
                enqueue(MailDismissOnboarding{});
            },
        };
    }

 private:
    static std::vector<OnboardingView::Feature> Highlights() {
        return {
            {ICON_LC_HOUSE, "The Start tab",
             "Browse the flowgraph gallery by category, keep favorites at hand, and reopen recent work.",
             "gallery_purple"},
            {ICON_LC_CAST, "Remote sharing",
             "Stream the whole interface to any browser on your network and invite viewers with a QR code.",
             "gallery_cyan"},
            {ICON_LC_SQUARE_TERMINAL, "Terminal tabs",
             "Open a shell as a tab next to your flowgraphs, with search and clickable links.", "gallery_green"},
            {ICON_LC_BOOK_OPEN, "Docs inside the app",
             "Read the documentation in its own tab and jump to any section from the docs search.",
             "gallery_orange"},
        };
    }

    static std::vector<OnboardingView::Section> Sections() {
        return {
            {"New Features", "gallery_green", Highlights()},
            {"Improvements",
             "gallery_cyan",
             {
                 {ICON_LC_ZAP, "A native interface",
                  "Every screen was rebuilt to draw natively and redraw only when something changes."},
                 {ICON_LC_SETTINGS, "Redesigned settings",
                  "Settings live in their own tab with card pages and refined interface scaling."},
             }},
        };
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_ONBOARDING_HH
