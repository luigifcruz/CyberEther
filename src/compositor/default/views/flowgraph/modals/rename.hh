#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_RENAME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_RENAME_HH

#include "../../components/modal_header.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <string>

namespace Jetstream {

struct RenameBlockView {
    struct Config {
        std::string oldName;
        Sakura::Callback<void(const std::string&)> onRename;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    bool canSubmit() const;
    void submit() const;

    Config config;
    std::string newName;
    ModalHeader header;
    Sakura::TextInput input;
    Sakura::Divider divider;
    Sakura::Button renameButton;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_MODALS_RENAME_HH
