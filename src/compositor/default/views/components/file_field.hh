#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FILE_FIELD_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FILE_FIELD_HH

#include "compact_button.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct FileField : public Sakura::Stack {
    struct Config {
        std::string icon = ICON_LC_FILE;
        std::string value;
        std::string hint;
        Sakura::TextInput::Submit submit = Sakura::TextInput::Submit::OnCommit;
        Sakura::Callback<void(const std::string&)> onChange;
        Sakura::Callback<void()> onBrowse;
    };

    void update(Config config);

 private:
    Sakura::PillInput input;
    CompactButton browse;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_FILE_FIELD_HH
