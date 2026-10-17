#ifndef JETSTREAM_RENDER_SAKURA_CLIPBOARD_HH
#define JETSTREAM_RENDER_SAKURA_CLIPBOARD_HH

#include <jetstream/macros.hh>

#include <string>

namespace Jetstream::Sakura {

void SetClipboardText(const std::string& value);
JETSTREAM_API std::string GetClipboardText();

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CLIPBOARD_HH
