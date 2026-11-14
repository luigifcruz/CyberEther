#ifndef JETSTREAM_RENDER_SAKURA_CLIPBOARD_HH
#define JETSTREAM_RENDER_SAKURA_CLIPBOARD_HH

#include <jetstream/macros.hh>

#include <string>

namespace Jetstream::Render {
class Window;
}  // namespace Jetstream::Render

namespace Jetstream::Sakura {

void SetClipboardText(Render::Window* window, const std::string& value);
JETSTREAM_API std::string GetClipboardText(Render::Window* window);

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CLIPBOARD_HH
