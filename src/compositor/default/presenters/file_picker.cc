#include "file_picker.hh"

namespace Jetstream {

FilePickerPresenter::FilePickerPresenter(const PresenterContext& context) : context(context) {}

FilePickerView::Config FilePickerPresenter::build() const {
    const auto& picker = context.state.filePicker;
    return {
        .active = picker.active,
        .overwritePending = picker.overwritePending,
        .generation = picker.generation,
        .mode = picker.mode,
        .root = picker.root,
        .directory = picker.directory,
        .selectedPath = picker.selectedPath,
        .filename = picker.filename,
        .error = picker.error,
        .extensions = picker.extensions,
        .entries = picker.entries,
        .onNavigate = [enqueue = context.callbacks.enqueueMail](U64 generation, std::string path) mutable {
            enqueue(MailFilePickerNavigate{generation, std::move(path)});
        },
        .onSelect = [enqueue = context.callbacks.enqueueMail](U64 generation, std::string path) mutable {
            enqueue(MailFilePickerSelect{generation, std::move(path)});
        },
        .onFilename = [enqueue = context.callbacks.enqueueMail](U64 generation, std::string value) mutable {
            enqueue(MailFilePickerSetFilename{generation, std::move(value)});
        },
        .onConfirm = [enqueue = context.callbacks.enqueueMail](U64 generation) mutable {
            enqueue(MailFilePickerConfirm{generation});
        },
        .onCancel = [enqueue = context.callbacks.enqueueMail](U64 generation) mutable {
            enqueue(MailFilePickerCancel{generation});
        },
    };
}

}  // namespace Jetstream
