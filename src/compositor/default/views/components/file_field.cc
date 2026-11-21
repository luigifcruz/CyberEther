#include "file_field.hh"

namespace Jetstream {

void FileField::update(Config config) {
    input.update({
        .icon = std::move(config.icon),
        .value = std::move(config.value),
        .hint = std::move(config.hint),
        .submit = config.submit,
        .width = 218.0f,
        .onChange = std::move(config.onChange),
    });
    browse.update({
        .icon = ICON_LC_FOLDER_OPEN,
        .onClick = std::move(config.onBrowse),
    });
    Stack::update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &input, .grow = 1.0f, .shrink = 1.0f},
            {.child = &browse},
        },
    });
}

}  // namespace Jetstream
