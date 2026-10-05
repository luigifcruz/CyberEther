#ifndef JETSTREAM_RENDER_COLORMAP_HH
#define JETSTREAM_RENDER_COLORMAP_HH

#include <array>
#include <memory>
#include <string>

#include "jetstream/parser.hh"
#include "jetstream/types.hh"
#include "jetstream/render/base/texture.hh"
#include "jetstream/render/base/window.hh"

namespace Jetstream::Render {

class JETSTREAM_API Colormap {
 public:
    static bool Valid(const std::string& name);
    static Parser::Map Format();

    Result create(const std::shared_ptr<Window>& window, const std::string& name);
    Result update(const std::string& name);

    const std::shared_ptr<Texture>& texture() const {
        return lutTexture;
    }

 private:
    std::string current;
    std::array<U8, 256 * 4> bytes{};
    std::shared_ptr<Texture> lutTexture;

    void fill(const std::string& name);
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_COLORMAP_HH
