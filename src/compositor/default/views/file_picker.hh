#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FILE_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FILE_PICKER_HH

#include "../model/file_picker.hh"
#include "components/modal_header.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <algorithm>
#include <filesystem>
#include <functional>
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

        modal.update({
            .id = "FilePickerModal",
            .size = Extent2D<F32>{720.0f, 0.0f},
            .onClose = [this]() {
                if (this->config.onCancel) {
                    this->config.onCancel(this->config.generation);
                }
            },
        });

        std::string description = save
            ? "Select where to save on the server."
            : "Choose a file from the server to open.";
        if (!save && !this->config.extensions.empty()) {
            description = "Choose a file from the server to open (" + extensionList() + ").";
        }
        header.update({
            .id = "FilePickerHeader",
            .title = save ? ICON_FA_FLOPPY_DISK " Save to Server"
                          : ICON_FA_FOLDER_OPEN " Open from Server",
            .description = description,
        });

        updateBreadcrumbs();

        fileGrid.update({
            .id = "FilePickerGrid",
            .columns = this->config.entries.empty() ? U64{1} : U64{2},
            .size = {0.0f, 300.0f},
            .cellPadding = {4.0f, 2.0f},
            .outerPadding = false,
        });
        emptySpacing.update({
            .id = "FilePickerEmptySpacing",
            .lines = 3,
        });
        emptyText.update({
            .id = "FilePickerEmpty",
            .str = this->config.extensions.empty() || save
                ? "This folder is empty."
                : "This folder has no matching files.",
            .tone = Sakura::Text::Tone::Disabled,
            .align = Sakura::Text::Align::Center,
        });
        entryItems.resize(this->config.entries.size());
        for (U64 i = 0; i < entryItems.size(); ++i) {
            const auto& entry = this->config.entries[i];
            entryItems[i].update({
                .id = "FilePickerEntry" + entry.path,
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
        }

        listDivider.update({
            .id = "FilePickerListDivider",
        });
        errorText.update({
            .id = "FilePickerError",
            .str = this->config.error,
            .colorKey = "error_red",
            .wrapped = true,
        });
        overwriteText.update({
            .id = "FilePickerOverwrite",
            .str = "A file with this name already exists. Click Overwrite to replace it.",
            .tone = Sakura::Text::Tone::Warning,
            .wrapped = true,
        });

        bool canConfirm = false;
        if (save) {
            canConfirm = !filename.empty();
            filenameContainer.update({
                .id = "FilePickerFilenameBox",
                .padding = 8.0f,
                .rounding = 10.0f,
                .border = true,
                .scrollbar = false,
                .mouseScroll = false,
                .colorKey = "card",
                .borderColorKey = "border",
            });
            filenameInput.update({
                .id = "FilePickerFilename",
                .value = filename,
                .hint = this->config.extensions.empty()
                    ? "Filename"
                    : "Filename (" + extensionList() + ")",
                .submit = Sakura::TextInput::Submit::OnEdit,
                .focusOutline = false,
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
        } else {
            const auto selected = std::find_if(this->config.entries.begin(), this->config.entries.end(),
                                               [this](const auto& entry) {
                                                   return entry.path == this->config.selectedPath;
                                               });
            canConfirm = selected != this->config.entries.end() && !selected->directory;
            selectionText.update({
                .id = "FilePickerSelection",
                .str = canConfirm ? std::string(ICON_FA_FILE_LINES " ") + selected->name
                                  : "No file selected.",
                .tone = canConfirm ? Sakura::Text::Tone::Secondary : Sakura::Text::Tone::Disabled,
            });
        }

        confirmButton.update({
            .id = "FilePickerConfirm",
            .str = !save ? ICON_FA_FOLDER_OPEN " Open"
                         : (this->config.overwritePending ? ICON_FA_TRIANGLE_EXCLAMATION " Overwrite"
                                                          : ICON_FA_FLOPPY_DISK " Save"),
            .size = {-1.0f, 40.0f},
            .variant = this->config.overwritePending ? Sakura::Button::Variant::Destructive
                                                     : Sakura::Button::Variant::Action,
            .disabled = !canConfirm,
            .onClick = [this]() {
                if (this->config.onConfirm) {
                    this->config.onConfirm(this->config.generation);
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) {
        if (!config.active) {
            return;
        }

        modal.render(ctx, [this](const Sakura::Context& ctx) {
            header.render(ctx);
            Sakura::HStack::Children breadcrumbChildren;
            breadcrumbChildren.reserve(crumbs.size() * 2);
            for (U64 i = 0; i < crumbs.size(); ++i) {
                if (i != 0) {
                    breadcrumbChildren.emplace_back([this](const Sakura::Context& ctx) {
                        crumbSeparator.render(ctx);
                    });
                }
                breadcrumbChildren.emplace_back([this, i](const Sakura::Context& ctx) {
                    if (crumbs[i].clickable) {
                        crumbButtons[i].render(ctx);
                    } else {
                        crumbTexts[i].render(ctx);
                    }
                });
            }
            crumbRow.render(ctx, std::move(breadcrumbChildren));
            Sakura::Grid::Children children;
            if (config.entries.empty()) {
                children.emplace_back([this](const Sakura::Context& ctx) {
                    emptySpacing.render(ctx);
                    emptyText.render(ctx);
                });
            } else {
                children.reserve(entryItems.size());
                for (const auto& item : entryItems) {
                    children.emplace_back([&item](const Sakura::Context& ctx) {
                        item.render(ctx);
                    });
                }
            }
            fileGrid.render(ctx, std::move(children));
            listDivider.render(ctx);
            if (!config.error.empty()) {
                errorText.render(ctx);
            } else if (config.overwritePending) {
                overwriteText.render(ctx);
            }
            if (config.mode == FilePickerMode::Save) {
                filenameContainer.render(ctx, [this](const Sakura::Context& ctx) {
                    filenameInput.render(ctx);
                });
            } else {
                selectionText.render(ctx);
            }
            confirmButton.render(ctx);
        });
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

        crumbRow.update({
            .id = "FilePickerCrumbs",
            .spacing = -4.0f,
        });
        crumbSeparator.update({
            .id = "FilePickerCrumbSeparator",
            .str = "/",
            .tone = Sakura::Text::Tone::Disabled,
        });
        crumbButtons.resize(crumbs.size());
        crumbTexts.resize(crumbs.size());
        for (U64 i = 0; i < crumbs.size(); ++i) {
            const auto& crumb = crumbs[i];
            if (crumb.clickable) {
                crumbButtons[i].update({
                    .id = "FilePickerCrumb" + std::to_string(i),
                    .str = crumb.label,
                    .variant = Sakura::Button::Variant::Text,
                    .textColorKey = "text_secondary",
                    .onClick = [this, path = crumb.path]() {
                        if (this->config.onNavigate) {
                            this->config.onNavigate(this->config.generation, path);
                        }
                    },
                });
            } else {
                crumbTexts[i].update({
                    .id = "FilePickerCrumb" + std::to_string(i),
                    .str = crumb.label,
                    .font = Sakura::Text::Font::Bold,
                });
            }
        }
    }

    Config config;
    std::string filename;
    bool filenameDirty = false;
    Sakura::Modal modal;
    ModalHeader header;
    std::vector<Crumb> crumbs;
    Sakura::HStack crumbRow;
    Sakura::Text crumbSeparator;
    std::vector<Sakura::Button> crumbButtons;
    std::vector<Sakura::Text> crumbTexts;
    Sakura::Grid fileGrid;
    Sakura::Spacing emptySpacing;
    Sakura::Text emptyText;
    std::vector<Sakura::NavigationItem> entryItems;
    Sakura::Divider listDivider;
    Sakura::Text errorText;
    Sakura::Text overwriteText;
    Sakura::Div filenameContainer;
    Sakura::TextInput filenameInput;
    Sakura::Text selectionText;
    Sakura::Button confirmButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FILE_PICKER_HH
