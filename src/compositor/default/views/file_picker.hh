#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FILE_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FILE_PICKER_HH

#include "../model/file_picker.hh"
#include "components/modal_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <algorithm>
#include <filesystem>
#include <functional>
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
        std::function<void(U64, std::string)> onNavigate;
        std::function<void(U64, std::string)> onSelect;
        std::function<void(U64, std::string)> onFilename;
        std::function<void(U64)> onConfirm;
        std::function<void(U64)> onCancel;
    };

    void update(Config config) {
        using Text = Sakura::Text;
        using Button = Sakura::Button;

        const bool changed = this->config.generation != config.generation;
        this->config = std::move(config);

        if (changed) {
            filename = this->config.filename;
            filenameDirty = false;
        } else if (filenameDirty) {
            if (this->config.filename == filename) {
                filenameDirty = false;
            }
        } else if (this->config.filename != filename) {
            filename = this->config.filename;
        }

        if (!this->config.active) {
            return;
        }

        const bool save = this->config.mode == FilePickerMode::Save;

        std::string description = save
            ? "Select where to save on the server."
            : "Choose a file from the server to open.";
        if (!save && !this->config.extensions.empty()) {
            description = "Choose a file from the server to open (" + extensionList() + ").";
        }
        header.update({
            .title = save ? ICON_FA_FLOPPY_DISK " Save to Server"
                          : ICON_FA_FOLDER_OPEN " Open from Server",
            .description = description,
        });

        updateBreadcrumbs();

        emptySpacing.update({.lines = 3});
        emptyText.update({
            .str = this->config.extensions.empty() || save
                ? "This folder is empty."
                : "This folder has no matching files.",
            .tone = Text::Tone::Disabled,
            .align = Text::Align::Center,
        });
        emptyCell.update({
            .items = {
                {.child = &emptySpacing},
                {.child = &emptyText},
            },
        });

        while (entryItems.size() < this->config.entries.size()) {
            entryItems.push_back(std::make_unique<Sakura::NavigationItem>());
        }
        std::vector<Sakura::Component*> gridItems;
        if (this->config.entries.empty()) {
            gridItems.push_back(&emptyCell);
        }
        for (U64 i = 0; i < this->config.entries.size(); ++i) {
            const auto& entry = this->config.entries[i];
            entryItems[i]->update({
                .label = std::string(entry.directory ? ICON_FA_FOLDER " " : ICON_FA_FILE_LINES " ") + entry.name,
                .selected = this->config.selectedPath == entry.path,
                .onSelect = [this, i]() {
                    if (i < this->config.entries.size() && this->config.onSelect) {
                        this->config.onSelect(this->config.generation, this->config.entries[i].path);
                    }
                },
                .onDoubleClick = [this, i]() {
                    if (i >= this->config.entries.size()) {
                        return;
                    }
                    const auto& entry = this->config.entries[i];
                    if (entry.directory) {
                        if (this->config.onNavigate) {
                            this->config.onNavigate(this->config.generation, entry.path);
                        }
                        return;
                    }
                    if (this->config.onSelect) {
                        this->config.onSelect(this->config.generation, entry.path);
                    }
                    if (this->config.mode == FilePickerMode::Open && this->config.onConfirm) {
                        this->config.onConfirm(this->config.generation);
                    }
                },
            });
            gridItems.push_back(entryItems[i].get());
        }
        fileGrid.update({
            .columns = this->config.entries.empty() ? U64{1} : U64{2},
            .size = {Sakura::Fill, 300.0f},
            .cellPadding = {4.0f, 2.0f},
            .outerPadding = false,
            .items = std::move(gridItems),
        });

        listDivider.update({});
        errorText.update({
            .str = this->config.error,
            .colorKey = "error_red",
            .wrapped = true,
        });
        overwriteText.update({
            .str = "A file with this name already exists. Click Overwrite to replace it.",
            .tone = Text::Tone::Warning,
            .wrapped = true,
        });

        bool canConfirm = false;
        if (save) {
            canConfirm = !filename.empty();
            filenameInput.update({
                .value = filename,
                .hint = this->config.extensions.empty()
                    ? "Filename"
                    : "Filename (" + extensionList() + ")",
                .submit = Sakura::TextInput::Submit::OnEdit,
                .onChange = [this](const std::string& value) {
                    filename = value;
                    filenameDirty = true;
                    if (this->config.onFilename) {
                        this->config.onFilename(this->config.generation, value);
                    }
                },
                .onSubmit = [this](const std::string& value) {
                    filename = value;
                    filenameDirty = true;
                    if (this->config.onFilename) {
                        this->config.onFilename(this->config.generation, value);
                    }
                    if (this->config.onConfirm) {
                        this->config.onConfirm(this->config.generation);
                    }
                },
            });
            filenameContainer.update({
                .content = &filenameInput,
                .padding = 8.0f,
                .rounding = 10.0f,
                .border = true,
                .colorKey = "card",
                .borderColorKey = "border",
            });
        } else {
            const auto selected = std::find_if(this->config.entries.begin(), this->config.entries.end(),
                                               [this](const auto& entry) {
                                                   return entry.path == this->config.selectedPath;
                                               });
            canConfirm = selected != this->config.entries.end() && !selected->directory;
            selectionText.update({
                .str = canConfirm ? std::string(ICON_FA_FILE_LINES " ") + selected->name
                                  : "No file selected.",
                .tone = canConfirm ? Text::Tone::Secondary : Text::Tone::Disabled,
            });
        }

        confirmButton.update({
            .str = !save ? ICON_FA_FOLDER_OPEN " Open"
                         : (this->config.overwritePending ? ICON_FA_TRIANGLE_EXCLAMATION " Overwrite"
                                                          : ICON_FA_FLOPPY_DISK " Save"),
            .size = {Sakura::Fill, 40.0f},
            .variant = this->config.overwritePending ? Button::Variant::Destructive
                                                     : Button::Variant::Action,
            .disabled = !canConfirm,
            .onClick = [this]() {
                if (this->config.onConfirm) {
                    this->config.onConfirm(this->config.generation);
                }
            },
        });

        std::vector<Sakura::Stack::Item> items = {
            {.child = &header},
            {.child = &crumbRow},
            {.child = &fileGrid},
            {.child = &listDivider},
        };
        if (!this->config.error.empty()) {
            items.push_back({.child = &errorText});
        } else if (this->config.overwritePending) {
            items.push_back({.child = &overwriteText});
        }
        items.push_back({.child = save ? static_cast<Sakura::Component*>(&filenameContainer) : &selectionText});
        items.push_back({.child = &confirmButton});
        stack.update({.items = std::move(items)});
        panel.update({
            .id = "FilePicker",
            .content = &stack,
            .backgroundColorKey = "popup_bg",
        });
    }

    Sakura::ModalLayer::Config layer() {
        if (!config.active) {
            return {};
        }
        return {
            .content = &panel,
            .size = Extent2D<F32>{720.0f, 0.0f},
            .onClose = [this]() {
                if (config.onCancel) {
                    config.onCancel(config.generation);
                }
            },
        };
    }

  private:
    struct Crumb {
        std::string label;
        std::string path;
        bool clickable = false;
    };

    std::string extensionList() const {
        std::string list;
        for (const auto& extension : config.extensions) {
            if (extension.empty()) {
                continue;
            }
            if (!list.empty()) {
                list += ", ";
            }
            list += extension.front() == '.' ? extension : "." + extension;
        }
        return list;
    }

    void updateBreadcrumbs() {
        const auto root = std::filesystem::path(config.root);
        const auto rootName = root.filename().string();

        std::vector<Crumb> full;
        full.push_back({rootName.empty() ? config.root : rootName, config.root, true});
        const auto relative = std::filesystem::path(config.directory).lexically_relative(root);
        auto accumulated = root;
        for (const auto& component : relative) {
            if (component == "." || component == "..") {
                continue;
            }
            accumulated /= component;
            full.push_back({component.string(), accumulated.string(), true});
        }

        crumbs.clear();
        if (full.size() > 4) {
            crumbs.push_back(full.front());
            crumbs.push_back({"…", full[full.size() - 3].path, true});
            crumbs.push_back(full[full.size() - 2]);
            crumbs.push_back(full.back());
        } else {
            crumbs = std::move(full);
        }
        crumbs.back().clickable = false;

        while (crumbButtons.size() < crumbs.size()) {
            crumbButtons.push_back(std::make_unique<Sakura::Button>());
            crumbTexts.push_back(std::make_unique<Sakura::Text>());
            crumbSeparators.push_back(std::make_unique<Sakura::Text>());
        }
        std::vector<Sakura::Stack::Item> crumbItems;
        for (U64 i = 0; i < crumbs.size(); ++i) {
            const auto& crumb = crumbs[i];
            if (i != 0) {
                crumbSeparators[i]->update({
                    .str = "/",
                    .tone = Sakura::Text::Tone::Disabled,
                });
                crumbItems.push_back({.child = crumbSeparators[i].get()});
            }
            if (crumb.clickable) {
                crumbButtons[i]->update({
                    .str = crumb.label,
                    .variant = Sakura::Button::Variant::Text,
                    .textColorKey = "text_secondary",
                    .onClick = [this, path = crumb.path]() {
                        if (this->config.onNavigate) {
                            this->config.onNavigate(this->config.generation, path);
                        }
                    },
                });
                crumbItems.push_back({.child = crumbButtons[i].get()});
            } else {
                crumbTexts[i]->update({
                    .str = crumb.label,
                    .font = Sakura::Text::Font::Bold,
                });
                crumbItems.push_back({.child = crumbTexts[i].get()});
            }
        }
        crumbRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 4.0f,
            .items = std::move(crumbItems),
        });
    }

    Config config;
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
