#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FILE_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FILE_PICKER_HH

#include "../model/file_picker.hh"
#include "components/modal_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FilePickerView {
    struct Config {
        bool active = false;
        bool overwritePending = false;
        U64 generation = 0;
        FilePickerMode mode = FilePickerMode::Open;
        std::string root;
        std::string directory;
        std::string selectedPath;
        std::string filename;
        std::string error;
        std::vector<std::string> extensions;
        std::vector<FilePickerEntry> entries;
        Sakura::Callback<void(U64, std::string)> onNavigate;
        Sakura::Callback<void(U64, std::string)> onSelect;
        Sakura::Callback<void(U64, std::string)> onFilename;
        Sakura::Callback<void(U64)> onConfirm;
        Sakura::Callback<void(U64)> onCancel;
    };

    void update(Config config);
    Sakura::ModalLayer::Config layer();

  private:
    struct Crumb {
        std::string label;
        std::string path;
        bool clickable = false;
    };

    std::string extensionList() const;
    void updateBreadcrumbs();    Config config;
    std::string filename;
    bool filenameDirty = false;
    ModalHeader header;
    std::vector<Crumb> crumbs;
    std::vector<std::unique_ptr<Sakura::Button>> crumbButtons;
    std::vector<std::unique_ptr<Sakura::Text>> crumbTexts;
    std::vector<std::unique_ptr<Sakura::Text>> crumbSeparators;
    Sakura::Stack crumbRow;
    Sakura::Spacer emptySpacing;
    Sakura::Text emptyText;
    Sakura::Stack emptyCell;
    std::vector<std::unique_ptr<Sakura::NavigationItem>> entryItems;
    Sakura::Grid fileGrid;
    Sakura::Divider listDivider;
    Sakura::Text errorText;
    Sakura::Text overwriteText;
    Sakura::TextInput filenameInput;
    Sakura::Div filenameContainer;
    Sakura::Text selectionText;
    Sakura::Button confirmButton;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FILE_PICKER_HH
