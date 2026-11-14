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

    explicit OnboardingModalPresenter(const PresenterContext& context);

    static constexpr const char* Release = "2.0";

    OnboardingView::Config build() const;

 private:
    static std::vector<OnboardingView::Feature> Basics();
    static std::vector<OnboardingView::Feature> Highlights();
    static std::vector<OnboardingView::Section> Sections();
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_MODAL_ONBOARDING_HH
