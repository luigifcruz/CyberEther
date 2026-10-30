#ifndef JETSTREAM_RENDER_WEBGPU_PRESENT_BINDINGS_HH
#define JETSTREAM_RENDER_WEBGPU_PRESENT_BINDINGS_HH

#include "jetstream/render/base/texture.hh"
#include "jetstream/types.hh"

#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>

namespace Jetstream::Render {

template<typename Binding>
class PresentBindingCache {
 public:
    using Create = std::function<Binding(const Texture&)>;
    using Release = std::function<void(Binding)>;

    PresentBindingCache(Create create, Release release) : create(std::move(create)), release(std::move(release)) {}

    ~PresentBindingCache() {
        clear();
    }

    PresentBindingCache(const PresentBindingCache&) = delete;
    PresentBindingCache& operator=(const PresentBindingCache&) = delete;

    Binding acquire(const std::shared_ptr<const Texture>& texture) {
        auto& entry = entries[texture.get()];
        if (entry.binding && (entry.owner.expired() || entry.owner.lock() != texture ||
                              entry.allocation != texture->allocation())) {
            release(entry.binding);
            entry.binding = Binding{};
        }
        if (!entry.binding) {
            entry.binding = create(*texture);
            entry.owner = texture;
            entry.allocation = texture->allocation();
        }
        entry.frame = frame;
        return entry.binding;
    }

    void advance() {
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->second.frame + 1 < frame || it->second.owner.expired()) {
                if (it->second.binding) {
                    release(it->second.binding);
                }
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
        ++frame;
    }

    void clear() {
        for (auto& [_, entry] : entries) {
            if (entry.binding) {
                release(entry.binding);
            }
        }
        entries.clear();
    }

    U64 size() const {
        return entries.size();
    }

 private:
    struct Entry {
        std::weak_ptr<const Texture> owner;
        U64 allocation = 0;
        U64 frame = 0;
        Binding binding{};
    };

    Create create;
    Release release;
    U64 frame = 1;
    std::unordered_map<const Texture*, Entry> entries;
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_WEBGPU_PRESENT_BINDINGS_HH
