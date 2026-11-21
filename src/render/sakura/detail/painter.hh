#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_PAINTER_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_PAINTER_HH

#include <jetstream/render/base/surface.hh>
#include <jetstream/render/base/window.hh>
#include <jetstream/render/components/font.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace Jetstream::Sakura {

struct Painter {
    enum class Kind : U8 {
        Shapes,
        Glyphs,
        Meshes,
        Curves,
    };

    struct Need {
        U64 vertices = 0;
        U64 indices = 0;
        U64 instances = 0;

        bool operator==(const Need&) const = default;
    };

    struct Lease {
        Kind kind = Kind::Shapes;
        U64 chunk = 0;
        U64 vertex = 0;
        U64 index = 0;
        U64 instance = 0;
        Need size;
    };

    struct Usage {
        U64 buffers = 0;
        U64 bytes = 0;
        U64 uploaded = 0;
    };

    struct Slot {
        Slot() = default;
        ~Slot();

        Slot(const Slot&) = delete;
        Slot& operator=(const Slot&) = delete;

        bool reserve(Painter& painter,
                     Kind kind,
                     const Need& need,
                     const std::shared_ptr<Render::Components::Font>& font = nullptr);
        void reset();

        bool held() const {
            return lease.has_value();
        }

        const Lease& operator*() const {
            return *lease;
        }

        const Lease* operator->() const {
            return &*lease;
        }

     private:
        Painter* painter = nullptr;
        std::shared_ptr<Render::Components::Font> key;
        std::optional<Lease> lease;
        U64 asked = 0;
        Need askedNeed;
    };

    static constexpr U64 ShapeStride = 32;
    static constexpr U64 GlyphVertexStride = 12;
    static constexpr U64 GlyphInstanceStride = 8;
    static constexpr U64 MeshVertexStride = 8;
    static constexpr U64 MeshInstanceStride = 4;
    static constexpr U64 CurveStride = 20;
    static constexpr U64 CurveSegments = 1024;
    static constexpr U64 ShapeIndices = 24;

    Painter();
    ~Painter();

    Painter(const Painter&) = delete;
    Painter& operator=(const Painter&) = delete;

    void demand(Kind kind, const Need& need, const std::shared_ptr<Render::Components::Font>& font = nullptr);
    bool starved() const;
    Result grow(Render::Window* window, Render::Surface::Config& surface);
    Result destroy(Render::Window* window);
    std::vector<std::shared_ptr<Render::Program>> programs() const;

    std::optional<Lease> acquire(Kind kind,
                                 const Need& need,
                                 const std::shared_ptr<Render::Components::Font>& font = nullptr,
                                 bool record = true);
    U64 generation() const;
    void release(const Lease& lease);

    std::span<F32> vertices(const Lease& lease);
    std::span<U32> indices(const Lease& lease);
    std::span<F32> instances(const Lease& lease);
    void touch(const Lease& lease);
    void touchInstances(const Lease& lease);
    std::shared_ptr<Render::Program> program(const Lease& lease) const;

    Result frame(const Extent2D<U64>& framebufferSize);
    Result flush();
    Usage usage() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_PAINTER_HH
