#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_FILE_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_FILE_PICKER_HH

#include <functional>
#include <string>
#include <vector>

namespace Jetstream {

enum class FilePickerMode {
    Open,
    Save,
};

struct FilePickerRequest {
    FilePickerMode mode = FilePickerMode::Open;
    std::string initialPath;
    std::vector<std::string> extensions;
    std::function<void(std::string)> callback;
};

struct FilePickerEntry {
    std::string name;
    std::string path;
    bool directory = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_MODEL_FILE_PICKER_HH
