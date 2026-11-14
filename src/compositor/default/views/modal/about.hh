#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ABOUT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ABOUT_HH

#include "jetstream/render/sakura/base.hh"

#include "jetstream/config.hh"

#include <string>

namespace Jetstream {

struct AboutView {
    struct Config {};

    void update(Config);
    Sakura::Component& component();

 private:
    Sakura::Spacer topSpacing;
    Sakura::Spacer middleSpacing;
    Sakura::Text title;
    Sakura::Text slogan;
    Sakura::Text details;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_ABOUT_HH
