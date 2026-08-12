#include <jetstream/render/sakura/clipboard.hh>

#include "helpers.hh"

namespace Jetstream::Sakura {

void SetClipboardText(const std::string& value) {
    ImGui::SetClipboardText(value.c_str());
}

}  // namespace Jetstream::Sakura
