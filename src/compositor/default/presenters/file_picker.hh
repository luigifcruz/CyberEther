#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FILE_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FILE_PICKER_HH

#include "context.hh"

#include "../model/messages.hh"
#include "../views/file_picker.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct FilePickerPresenter {
    const PresenterContext& context;

    explicit FilePickerPresenter(const PresenterContext& context);
    FilePickerView::Config build() const;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FILE_PICKER_HH
