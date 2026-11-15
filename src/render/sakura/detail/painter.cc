#include "painter.hh"

#include <jetstream/render/base.hh>
#include <jetstream/render/components/generic.hh>

#include "resources/shaders/global_shaders.hh"

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr U64 kUniformFloats = 8;
constexpr U64 kHeadroom = 2;

struct Layout {
    const char* shader = "";
    U64 vertexStride = 0;
    U64 instanceStride = 0;
    bool leasedVertices = false;
    bool leasedIndices = false;
    Painter::Need minimum;
};

Layout LayoutOf(Painter::Kind kind) {
    switch (kind) {
        case Painter::Kind::Shapes:
            return {
                .shader = "shapes",
                .vertexStride = 2,
                .instanceStride = Painter::ShapeStride,
                .minimum = {.instances = 256},
            };
        case Painter::Kind::Glyphs:
            return {
                .shader = "glyphs",
                .vertexStride = Painter::GlyphVertexStride,
                .instanceStride = Painter::GlyphInstanceStride,
                .leasedVertices = true,
                .minimum = {.vertices = 4096, .instances = 64},
            };
        case Painter::Kind::Meshes:
            return {
                .shader = "meshes",
                .vertexStride = Painter::MeshVertexStride,
                .instanceStride = Painter::MeshInstanceStride,
                .leasedVertices = true,
                .leasedIndices = true,
                .minimum = {.vertices = 4096, .indices = 12288, .instances = 64},
            };
        case Painter::Kind::Curves:
            return {
                .shader = "strokes",
                .vertexStride = 2,
                .instanceStride = Painter::CurveStride,
                .minimum = {.instances = 64},
            };
    }
    return {};
}

struct Allocator {
    struct Extent {
        U64 offset = 0;
        U64 size = 0;
    };

    U64 capacity = 0;
    std::vector<Extent> free;

    void reset(U64 size) {
        capacity = size;
        free.clear();
        if (size > 0) {
            free.push_back({0, size});
        }
    }

    bool fits(U64 size) const {
        return size == 0 || std::any_of(free.begin(), free.end(), [size](const Extent& extent) {
            return extent.size >= size;
        });
    }

    U64 available() const {
        U64 total = 0;
        for (const auto& extent : free) {
            total += extent.size;
        }
        return total;
    }

    U64 take(U64 size) {
        if (size == 0) {
            return 0;
        }
        for (auto it = free.begin(); it != free.end(); ++it) {
            if (it->size < size) {
                continue;
            }
            const U64 offset = it->offset;
            it->offset += size;
            it->size -= size;
            if (it->size == 0) {
                free.erase(it);
            }
            return offset;
        }
        return capacity;
    }

    void give(U64 offset, U64 size) {
        if (size == 0) {
            return;
        }
        auto next = std::lower_bound(free.begin(), free.end(), offset, [](const Extent& extent, U64 value) {
            return extent.offset < value;
        });
        next = free.insert(next, {offset, size});
        if (next + 1 != free.end() && next->offset + next->size == (next + 1)->offset) {
            next->size += (next + 1)->size;
            free.erase(next + 1);
        }
        if (next != free.begin() && (next - 1)->offset + (next - 1)->size == next->offset) {
            (next - 1)->size += next->size;
            free.erase(next);
        }
    }
};

struct Stream {
    U64 stride = 0;
    U64 capacity = 0;
    bool leased = false;
    Allocator allocator;
    U64 dirtyBegin = std::numeric_limits<U64>::max();
    U64 dirtyEnd = 0;
    std::shared_ptr<Render::Buffer> buffer;

    void touch(U64 offset, U64 size) {
        if (size == 0) {
            return;
        }
        dirtyBegin = std::min(dirtyBegin, offset);
        dirtyEnd = std::max(dirtyEnd, offset + size);
    }

    Result flush(U64& uploaded) {
        if (!buffer || dirtyEnd <= dirtyBegin) {
            return Result::SUCCESS;
        }
        const U64 begin = dirtyBegin;
        const U64 end = std::min(dirtyEnd, capacity);
        if (end > begin) {
            JST_CHECK(buffer->update(begin, end - begin));
            uploaded += (end - begin) * stride * sizeof(F32);
        }
        dirtyBegin = std::numeric_limits<U64>::max();
        dirtyEnd = 0;
        return Result::SUCCESS;
    }
};

struct Chunk : public Render::Components::Generic {
    Painter::Kind kind = Painter::Kind::Shapes;
    std::shared_ptr<Render::Components::Font> font;
    Painter::Need capacity;
    Layout layout;

    std::vector<F32> vertexData;
    std::vector<U32> indexData;
    std::vector<F32> instanceData;
    std::array<F32, kUniformFloats> uniforms = {};
    bool uniformsDirty = true;

    Stream vertices;
    Stream indices;
    Stream instances;

    std::shared_ptr<Render::Buffer> uniformBuffer;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;

    void fillTemplates() {
        if (kind == Painter::Kind::Shapes) {
            constexpr F32 corners[16] = {
                -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
                -2.0f, -2.0f, 2.0f, -2.0f, 2.0f, 2.0f, -2.0f, 2.0f,
            };
            constexpr U32 frame[Painter::ShapeIndices] = {
                0, 1, 5, 0, 5, 4,
                1, 2, 6, 1, 6, 5,
                2, 3, 7, 2, 7, 6,
                3, 0, 4, 3, 4, 7,
            };
            vertexData.assign(corners, corners + 16);
            indexData.assign(frame, frame + Painter::ShapeIndices);
        } else if (kind == Painter::Kind::Curves) {
            constexpr F32 sides[4] = {2.0f, 1.0f, -1.0f, -2.0f};
            const U64 points = Painter::CurveSegments + 1;
            vertexData.resize(points * 4 * 2);
            for (U64 point = 0; point < points; ++point) {
                for (U64 side = 0; side < 4; ++side) {
                    vertexData[(point * 4 + side) * 2 + 0] = static_cast<F32>(point);
                    vertexData[(point * 4 + side) * 2 + 1] = sides[side];
                }
            }
            indexData.resize(Painter::CurveSegments * 18);
            for (U64 segment = 0; segment < Painter::CurveSegments; ++segment) {
                const U32 a = static_cast<U32>(segment * 4);
                const U32 b = a + 4;
                const U32 triangles[18] = {
                    b + 1, a + 1, a + 2,
                    a + 2, b + 2, b + 1,
                    b + 1, a + 1, a + 0,
                    a + 0, b + 0, b + 1,
                    b + 2, a + 2, a + 3,
                    a + 3, b + 3, b + 2,
                };
                std::copy_n(triangles, 18, indexData.begin() + static_cast<I64>(segment * 18));
            }
        } else if (kind == Painter::Kind::Glyphs) {
            const U64 quads = capacity.vertices / 4;
            indexData.resize(quads * 6);
            for (U64 quad = 0; quad < quads; ++quad) {
                const U32 base = static_cast<U32>(quad * 4);
                const U32 pattern[6] = {base, base + 1, base + 2, base + 2, base + 3, base};
                std::copy_n(pattern, 6, indexData.begin() + static_cast<I64>(quad * 6));
            }
        }
    }

    Result create(Render::Window* window) override {
        layout = LayoutOf(kind);

        if (layout.leasedVertices) {
            vertexData.assign(capacity.vertices * layout.vertexStride, 0.0f);
        }
        if (layout.leasedIndices) {
            indexData.assign(capacity.indices, 0);
        }
        instanceData.assign(capacity.instances * layout.instanceStride, 0.0f);
        fillTemplates();

        vertices.stride = layout.vertexStride;
        vertices.capacity = vertexData.size() / layout.vertexStride;
        vertices.leased = layout.leasedVertices;
        vertices.allocator.reset(layout.leasedVertices ? vertices.capacity : 0);

        indices.stride = 1;
        indices.capacity = indexData.size();
        indices.leased = layout.leasedIndices;
        indices.allocator.reset(layout.leasedIndices ? indices.capacity : 0);

        instances.stride = layout.instanceStride;
        instances.capacity = capacity.instances;
        instances.leased = true;
        instances.allocator.reset(instances.capacity);

        {
            Render::Buffer::Config cfg;
            cfg.buffer = uniforms.data();
            cfg.elementByteSize = sizeof(F32) * kUniformFloats;
            cfg.size = 1;
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(uniformBuffer, cfg));
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = vertexData.data();
            cfg.elementByteSize = sizeof(F32) * layout.vertexStride;
            cfg.size = vertices.capacity;
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(vertices.buffer, cfg));
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = indexData.data();
            cfg.elementByteSize = sizeof(U32);
            cfg.size = indices.capacity;
            cfg.target = Render::Buffer::Target::VERTEX_INDICES;
            JST_CHECK(window->build(indices.buffer, cfg));
        }

        {
            Render::Buffer::Config cfg;
            cfg.buffer = instanceData.data();
            cfg.elementByteSize = sizeof(F32) * layout.instanceStride;
            cfg.size = instances.capacity;
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(instances.buffer, cfg));
        }

        {
            Render::Vertex::Config cfg;
            cfg.vertices = {
                {vertices.buffer, static_cast<U32>(layout.vertexStride)},
            };
            cfg.instances = {
                {instances.buffer, static_cast<U32>(layout.instanceStride)},
            };
            cfg.indices = indices.buffer;
            JST_CHECK(window->build(vertex, cfg));
        }

        {
            Render::Draw::Config cfg;
            cfg.numberOfDraws = 1;
            cfg.numberOfInstances = 1;
            cfg.buffer = vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            JST_CHECK(window->build(draw, cfg));
        }

        {
            Render::Program::Config cfg;
            cfg.shaders = GlobalShadersPackage[layout.shader];
            cfg.draws = {
                draw,
            };
            if (font) {
                cfg.textures = {
                    font->atlas(),
                };
            }
            cfg.buffers = {
                {uniformBuffer, Render::Program::Target::VERTEX | Render::Program::Target::FRAGMENT},
            };
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(program, cfg));
        }

        return Result::SUCCESS;
    }

    bool fits(const Painter::Need& need) const {
        return (!vertices.leased || vertices.allocator.fits(need.vertices)) &&
               (!indices.leased || indices.allocator.fits(need.indices)) &&
               instances.allocator.fits(need.instances);
    }

    Result flush(U64& uploaded) {
        if (uniformsDirty) {
            JST_CHECK(uniformBuffer->update());
            uniformsDirty = false;
        }
        if (vertices.leased) {
            JST_CHECK(vertices.flush(uploaded));
        }
        if (indices.leased) {
            JST_CHECK(indices.flush(uploaded));
        }
        return instances.flush(uploaded);
    }

    U64 bytes() const {
        return vertexData.size() * sizeof(F32) + indexData.size() * sizeof(U32) + instanceData.size() * sizeof(F32);
    }
};

U64 Grown(U64 need, U64 minimum, U64 previous) {
    if (need == 0 && minimum == 0) {
        return 0;
    }
    return std::bit_ceil(std::max({need * kHeadroom, minimum, previous * 2}));
}

}  // namespace

struct Painter::Impl {
    struct Pending {
        Kind kind = Kind::Shapes;
        std::shared_ptr<Render::Components::Font> font;
        Need need;
    };

    std::vector<std::shared_ptr<Chunk>> chunks;
    std::vector<Pending> pending;
    Extent2D<U64> framebufferSize = {0, 0};
    U64 generation = 1;
    U64 uploaded = 0;

    void applyFrame(Chunk& chunk) const {
        std::array<F32, kUniformFloats> next = {};
        const F32 width = static_cast<F32>(framebufferSize.x);
        const F32 height = static_cast<F32>(framebufferSize.y);
        switch (chunk.kind) {
            case Kind::Shapes:
                next[0] = framebufferSize.x > 0 ? 2.0f / width : 0.0f;
                next[1] = framebufferSize.y > 0 ? 2.0f / height : 0.0f;
                break;
            case Kind::Glyphs:
                next[0] = framebufferSize.x > 0 ? 2.0f / width : 0.0f;
                next[1] = framebufferSize.y > 0 ? 2.0f / height : 0.0f;
                next[2] = chunk.font ? chunk.font->atlasPixelRange() : 0.0f;
                next[3] = chunk.font ? chunk.font->getConfig().size * chunk.font->atlasScale() : 0.0f;
                break;
            case Kind::Meshes:
                next[0] = width;
                next[1] = height;
                break;
            case Kind::Curves:
                next[0] = width;
                next[1] = height;
                next[2] = static_cast<F32>(CurveSegments);
                break;
        }
        if (next != chunk.uniforms) {
            chunk.uniforms = next;
            chunk.uniformsDirty = true;
        }
    }
};

Painter::Slot::~Slot() {
    reset();
}

bool Painter::Slot::reserve(Painter& target,
                            Kind kind,
                            const Need& need,
                            const std::shared_ptr<Render::Components::Font>& font) {
    if (lease.has_value() && painter == &target && lease->kind == kind && key == font &&
        lease->size.vertices >= need.vertices && lease->size.indices >= need.indices &&
        lease->size.instances >= need.instances &&
        lease->size.vertices <= std::max<U64>(4 * need.vertices, 16) &&
        lease->size.indices <= std::max<U64>(4 * need.indices, 16) &&
        lease->size.instances <= std::max<U64>(4 * need.instances, 16)) {
        return true;
    }
    const Need rounded = {
        .vertices = need.vertices > 0 ? std::bit_ceil(need.vertices) : 0,
        .indices = need.indices > 0 ? std::bit_ceil(need.indices) : 0,
        .instances = need.instances > 0 ? std::bit_ceil(need.instances) : 0,
    };
    auto next = target.acquire(kind, rounded, font, false);
    if (!next.has_value()) {
        const U64 generation = target.generation();
        if (asked != generation || askedNeed != rounded) {
            target.demand(kind, rounded, font);
            asked = generation;
            askedNeed = rounded;
        }
        return false;
    }
    asked = 0;
    reset();
    painter = &target;
    key = font;
    lease = next;
    return true;
}

void Painter::Slot::reset() {
    if (painter && lease.has_value()) {
        painter->release(*lease);
    }
    painter = nullptr;
    key.reset();
    lease.reset();
}

Painter::Painter() {
    impl = std::make_unique<Impl>();
}

Painter::~Painter() = default;

void Painter::demand(Kind kind, const Need& need, const std::shared_ptr<Render::Components::Font>& font) {
    for (auto& entry : impl->pending) {
        if (entry.kind == kind && entry.font == font) {
            entry.need.vertices += need.vertices;
            entry.need.indices += need.indices;
            entry.need.instances += need.instances;
            return;
        }
    }
    impl->pending.push_back({kind, font, need});
}

bool Painter::starved() const {
    return !impl->pending.empty();
}

Result Painter::grow(Render::Window* window, Render::Surface::Config& surface) {
    auto pending = std::move(impl->pending);
    impl->pending.clear();
    ++impl->generation;
    for (const auto& entry : pending) {
        const Layout layout = LayoutOf(entry.kind);
        Need previous;
        bool satisfied = false;
        for (const auto& chunk : impl->chunks) {
            if (chunk->kind == entry.kind && chunk->font == entry.font) {
                satisfied = satisfied || chunk->fits(entry.need);
                previous.vertices = std::max(previous.vertices, chunk->capacity.vertices);
                previous.indices = std::max(previous.indices, chunk->capacity.indices);
                previous.instances = std::max(previous.instances, chunk->capacity.instances);
            }
        }
        if (satisfied) {
            continue;
        }
        auto chunk = std::make_shared<Chunk>();
        chunk->kind = entry.kind;
        chunk->font = entry.font;
        chunk->capacity = {
            .vertices = layout.leasedVertices
                            ? Grown(entry.need.vertices, layout.minimum.vertices, previous.vertices) : 0,
            .indices = layout.leasedIndices
                           ? Grown(entry.need.indices, layout.minimum.indices, previous.indices) : 0,
            .instances = Grown(entry.need.instances, layout.minimum.instances, previous.instances),
        };
        JST_CHECK(window->bind(std::static_pointer_cast<Render::Components::Generic>(chunk)));
        impl->applyFrame(*chunk);
        surface.programs.push_back(chunk->program);
        impl->chunks.push_back(std::move(chunk));
    }
    return Result::SUCCESS;
}

Result Painter::destroy(Render::Window* window) {
    for (auto& chunk : impl->chunks) {
        if (window) {
            JST_CHECK(window->unbind(std::static_pointer_cast<Render::Components::Generic>(chunk)));
        }
    }
    impl->chunks.clear();
    impl->pending.clear();
    ++impl->generation;
    return Result::SUCCESS;
}

std::vector<std::shared_ptr<Render::Program>> Painter::programs() const {
    std::vector<std::shared_ptr<Render::Program>> programs;
    programs.reserve(impl->chunks.size());
    for (const auto& chunk : impl->chunks) {
        programs.push_back(chunk->program);
    }
    return programs;
}

U64 Painter::generation() const {
    return impl->generation;
}

std::optional<Painter::Lease> Painter::acquire(Kind kind,
                                               const Need& need,
                                               const std::shared_ptr<Render::Components::Font>& font,
                                               bool record) {
    for (U64 index = 0; index < impl->chunks.size(); ++index) {
        auto& chunk = *impl->chunks[index];
        if (chunk.kind != kind || chunk.font != font || !chunk.fits(need)) {
            continue;
        }
        Lease lease = {.kind = kind, .chunk = index, .size = need};
        if (chunk.vertices.leased) {
            lease.vertex = chunk.vertices.allocator.take(need.vertices);
        } else {
            lease.size.vertices = 0;
        }
        if (chunk.indices.leased) {
            lease.index = chunk.indices.allocator.take(need.indices);
        } else {
            lease.size.indices = 0;
        }
        lease.instance = chunk.instances.allocator.take(need.instances);
        return lease;
    }
    if (record) {
        demand(kind, need, font);
    }
    return std::nullopt;
}

void Painter::release(const Lease& lease) {
    if (lease.chunk >= impl->chunks.size()) {
        return;
    }
    auto& chunk = *impl->chunks[lease.chunk];
    if (chunk.vertices.leased) {
        chunk.vertices.allocator.give(lease.vertex, lease.size.vertices);
    }
    if (chunk.indices.leased) {
        chunk.indices.allocator.give(lease.index, lease.size.indices);
    }
    chunk.instances.allocator.give(lease.instance, lease.size.instances);
}

std::span<F32> Painter::vertices(const Lease& lease) {
    auto& chunk = *impl->chunks[lease.chunk];
    return std::span<F32>(chunk.vertexData).subspan(lease.vertex * chunk.vertices.stride,
                                                   lease.size.vertices * chunk.vertices.stride);
}

std::span<U32> Painter::indices(const Lease& lease) {
    auto& chunk = *impl->chunks[lease.chunk];
    return std::span<U32>(chunk.indexData).subspan(lease.index, lease.size.indices);
}

std::span<F32> Painter::instances(const Lease& lease) {
    auto& chunk = *impl->chunks[lease.chunk];
    return std::span<F32>(chunk.instanceData).subspan(lease.instance * chunk.instances.stride,
                                                     lease.size.instances * chunk.instances.stride);
}

void Painter::touch(const Lease& lease) {
    auto& chunk = *impl->chunks[lease.chunk];
    if (chunk.vertices.leased) {
        chunk.vertices.touch(lease.vertex, lease.size.vertices);
    }
    if (chunk.indices.leased) {
        chunk.indices.touch(lease.index, lease.size.indices);
    }
    chunk.instances.touch(lease.instance, lease.size.instances);
}

void Painter::touchInstances(const Lease& lease) {
    auto& chunk = *impl->chunks[lease.chunk];
    chunk.instances.touch(lease.instance, lease.size.instances);
}

std::shared_ptr<Render::Program> Painter::program(const Lease& lease) const {
    return impl->chunks[lease.chunk]->program;
}

Result Painter::frame(const Extent2D<U64>& framebufferSize) {
    impl->framebufferSize = framebufferSize;
    for (auto& chunk : impl->chunks) {
        impl->applyFrame(*chunk);
    }
    return Result::SUCCESS;
}

Result Painter::flush() {
    for (auto& chunk : impl->chunks) {
        JST_CHECK(chunk->flush(impl->uploaded));
    }
    return Result::SUCCESS;
}

Painter::Usage Painter::usage() const {
    Usage usage = {.buffers = impl->chunks.size() * 4, .uploaded = impl->uploaded};
    for (const auto& chunk : impl->chunks) {
        usage.bytes += chunk->bytes();
    }
    return usage;
}

}  // namespace Jetstream::Sakura
