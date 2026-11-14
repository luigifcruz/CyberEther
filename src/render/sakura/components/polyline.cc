#include <jetstream/render/sakura/components/polyline.hh>

#include <jetstream/render/base.hh>
#include <jetstream/render/components/mesh.hh>

#include "../detail/drawable.hh"
#include "../detail/helpers.hh"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kFixNormalMaxInvLength2 = 100.0f;

void Normalize(F32& x, F32& y) {
    const F32 d2 = x * x + y * y;
    if (d2 > 0.0f) {
        const F32 inv = 1.0f / std::sqrt(d2);
        x *= inv;
        y *= inv;
    }
}

void FixNormal(F32& x, F32& y) {
    const F32 d2 = x * x + y * y;
    if (d2 > 0.000001f) {
        const F32 inv = std::min(1.0f / d2, kFixNormalMaxInvLength2);
        x *= inv;
        y *= inv;
    }
}

ColorRGBA<F32> Transparent(ColorRGBA<F32> color) {
    color.a = 0.0f;
    return color;
}

struct Geometry {
    std::vector<Extent2D<F32>> positions;
    std::vector<ColorRGBA<F32>> colors;
    std::vector<U32> indices;

    void clear() {
        positions.clear();
        colors.clear();
        indices.clear();
    }

    U32 base() const {
        return static_cast<U32>(positions.size());
    }

    void vertex(Extent2D<F32> position, ColorRGBA<F32> color) {
        positions.push_back(position);
        colors.push_back(color);
    }

    void triangle(U32 a, U32 b, U32 c) {
        indices.push_back(a);
        indices.push_back(b);
        indices.push_back(c);
    }

    void stroke(const Polyline::Instance& instance, F32 fringe, F32 minimumThickness) {
        const auto& points = instance.points;
        const U64 pointCount = points.size();
        if (pointCount < 2 || instance.color.a <= 0.0f) {
            return;
        }

        const bool closed = instance.closed;
        const U64 count = closed ? pointCount : pointCount - 1;
        const F32 thickness = std::max(instance.thickness, minimumThickness);
        const bool thick = thickness > fringe;
        const ColorRGBA<F32> color = instance.color;
        const ColorRGBA<F32> transparent = Transparent(color);

        std::vector<Extent2D<F32>> normals(pointCount);
        for (U64 i1 = 0; i1 < count; ++i1) {
            const U64 i2 = (i1 + 1) == pointCount ? 0 : i1 + 1;
            F32 dx = points[i2].x - points[i1].x;
            F32 dy = points[i2].y - points[i1].y;
            Normalize(dx, dy);
            normals[i1] = {dy, -dx};
        }
        if (!closed) {
            normals[pointCount - 1] = normals[pointCount - 2];
        }

        const U32 start = base();

        if (!thick) {
            const F32 halfDrawSize = fringe;
            std::vector<Extent2D<F32>> edges(pointCount * 2);
            if (!closed) {
                edges[0] = {points[0].x + normals[0].x * halfDrawSize, points[0].y + normals[0].y * halfDrawSize};
                edges[1] = {points[0].x - normals[0].x * halfDrawSize, points[0].y - normals[0].y * halfDrawSize};
                const U64 last = pointCount - 1;
                edges[last * 2 + 0] = {points[last].x + normals[last].x * halfDrawSize,
                                       points[last].y + normals[last].y * halfDrawSize};
                edges[last * 2 + 1] = {points[last].x - normals[last].x * halfDrawSize,
                                       points[last].y - normals[last].y * halfDrawSize};
            }

            U32 idx1 = start;
            for (U64 i1 = 0; i1 < count; ++i1) {
                const U64 i2 = (i1 + 1) == pointCount ? 0 : i1 + 1;
                const U32 idx2 = (i1 + 1) == pointCount ? start : idx1 + 3;

                F32 dmx = (normals[i1].x + normals[i2].x) * 0.5f;
                F32 dmy = (normals[i1].y + normals[i2].y) * 0.5f;
                FixNormal(dmx, dmy);
                dmx *= halfDrawSize;
                dmy *= halfDrawSize;

                edges[i2 * 2 + 0] = {points[i2].x + dmx, points[i2].y + dmy};
                edges[i2 * 2 + 1] = {points[i2].x - dmx, points[i2].y - dmy};

                triangle(idx2 + 0, idx1 + 0, idx1 + 2);
                triangle(idx1 + 2, idx2 + 2, idx2 + 0);
                triangle(idx2 + 1, idx1 + 1, idx1 + 0);
                triangle(idx1 + 0, idx2 + 0, idx2 + 1);

                idx1 = idx2;
            }

            for (U64 i = 0; i < pointCount; ++i) {
                vertex(points[i], color);
                vertex(edges[i * 2 + 0], transparent);
                vertex(edges[i * 2 + 1], transparent);
            }
            return;
        }

        const F32 halfInner = (thickness - fringe) * 0.5f;
        std::vector<Extent2D<F32>> edges(pointCount * 4);
        const auto setEdges = [&](U64 index, F32 nx, F32 ny, F32 outer, F32 inner) {
            edges[index * 4 + 0] = {points[index].x + nx * outer, points[index].y + ny * outer};
            edges[index * 4 + 1] = {points[index].x + nx * inner, points[index].y + ny * inner};
            edges[index * 4 + 2] = {points[index].x - nx * inner, points[index].y - ny * inner};
            edges[index * 4 + 3] = {points[index].x - nx * outer, points[index].y - ny * outer};
        };

        if (!closed) {
            const U64 last = pointCount - 1;
            setEdges(0, normals[0].x, normals[0].y, halfInner + fringe, halfInner);
            setEdges(last, normals[last].x, normals[last].y, halfInner + fringe, halfInner);
        }

        U32 idx1 = start;
        for (U64 i1 = 0; i1 < count; ++i1) {
            const U64 i2 = (i1 + 1) == pointCount ? 0 : i1 + 1;
            const U32 idx2 = (i1 + 1) == pointCount ? start : idx1 + 4;

            F32 dmx = (normals[i1].x + normals[i2].x) * 0.5f;
            F32 dmy = (normals[i1].y + normals[i2].y) * 0.5f;
            FixNormal(dmx, dmy);
            setEdges(i2, dmx, dmy, halfInner + fringe, halfInner);

            triangle(idx2 + 1, idx1 + 1, idx1 + 2);
            triangle(idx1 + 2, idx2 + 2, idx2 + 1);
            triangle(idx2 + 1, idx1 + 1, idx1 + 0);
            triangle(idx1 + 0, idx2 + 0, idx2 + 1);
            triangle(idx2 + 2, idx1 + 2, idx1 + 3);
            triangle(idx1 + 3, idx2 + 3, idx2 + 2);

            idx1 = idx2;
        }

        for (U64 i = 0; i < pointCount; ++i) {
            vertex(edges[i * 4 + 0], transparent);
            vertex(edges[i * 4 + 1], color);
            vertex(edges[i * 4 + 2], color);
            vertex(edges[i * 4 + 3], transparent);
        }
    }

    void fill(const Polyline::Instance& instance, F32 fringe) {
        const auto& points = instance.points;
        const U64 pointCount = points.size();
        if (pointCount < 3 || instance.color.a <= 0.0f) {
            return;
        }

        const ColorRGBA<F32> color = instance.color;
        const ColorRGBA<F32> transparent = Transparent(color);
        const U32 inner = base();
        const U32 outer = inner + 1;

        for (U64 i = 2; i < pointCount; ++i) {
            triangle(inner, inner + static_cast<U32>((i - 1) << 1), inner + static_cast<U32>(i << 1));
        }

        std::vector<Extent2D<F32>> normals(pointCount);
        for (U64 i0 = pointCount - 1, i1 = 0; i1 < pointCount; i0 = i1++) {
            F32 dx = points[i1].x - points[i0].x;
            F32 dy = points[i1].y - points[i0].y;
            Normalize(dx, dy);
            normals[i0] = {dy, -dx};
        }

        for (U64 i0 = pointCount - 1, i1 = 0; i1 < pointCount; i0 = i1++) {
            F32 dmx = (normals[i0].x + normals[i1].x) * 0.5f;
            F32 dmy = (normals[i0].y + normals[i1].y) * 0.5f;
            FixNormal(dmx, dmy);
            dmx *= fringe * 0.5f;
            dmy *= fringe * 0.5f;

            vertex({points[i1].x - dmx, points[i1].y - dmy}, color);
            vertex({points[i1].x + dmx, points[i1].y + dmy}, transparent);

            const U32 a = static_cast<U32>(i1 << 1);
            const U32 b = static_cast<U32>(i0 << 1);
            triangle(inner + a, inner + b, outer + b);
            triangle(outer + b, outer + a, inner + a);
        }
    }
};

U64 IndexCapacity(U64 vertexCapacity) {
    return (vertexCapacity * 9 + 1) / 2;
}

}  // namespace

std::vector<Extent2D<F32>> Polyline::Bezier(Extent2D<F32> p0,
                                            Extent2D<F32> p1,
                                            Extent2D<F32> p2,
                                            Extent2D<F32> p3,
                                            U64 segments) {
    segments = std::max<U64>(1, segments);
    std::vector<Extent2D<F32>> points;
    points.reserve(segments + 1);
    points.push_back(p0);
    const F32 step = 1.0f / static_cast<F32>(segments);
    for (U64 i = 1; i <= segments; ++i) {
        const F32 t = step * static_cast<F32>(i);
        const F32 u = 1.0f - t;
        const F32 w0 = u * u * u;
        const F32 w1 = 3.0f * u * u * t;
        const F32 w2 = 3.0f * u * t * t;
        const F32 w3 = t * t * t;
        points.push_back({
            w0 * p0.x + w1 * p1.x + w2 * p2.x + w3 * p3.x,
            w0 * p0.y + w1 * p1.y + w2 * p2.y + w3 * p3.y,
        });
    }
    return points;
}

std::vector<Extent2D<F32>> Polyline::Circle(Extent2D<F32> center, F32 radius, U64 segments) {
    segments = std::max<U64>(3, segments);
    std::vector<Extent2D<F32>> points;
    points.reserve(segments);
    const F32 maxAngle = 2.0f * std::numbers::pi_v<F32> * static_cast<F32>(segments - 1) / static_cast<F32>(segments);
    for (U64 i = 0; i < segments; ++i) {
        const F32 angle = maxAngle * static_cast<F32>(i) / static_cast<F32>(segments - 1);
        points.push_back({center.x + std::cos(angle) * radius, center.y + std::sin(angle) * radius});
    }
    return points;
}

std::vector<Polyline::Instance> Polyline::CornerMasks(Rect rect,
                                                      F32 radius,
                                                      ColorRGBA<F32> top,
                                                      ColorRGBA<F32> bottom,
                                                      U64 segments) {
    radius = std::min(radius, std::min(rect.width, rect.height) * 0.5f);
    if (radius <= 0.0f) {
        return {};
    }
    segments = std::max<U64>(1, segments);

    struct Corner {
        Extent2D<F32> point;
        Extent2D<F32> center;
        F32 start;
        ColorRGBA<F32> color;
    };
    const F32 x0 = rect.x;
    const F32 y0 = rect.y;
    const F32 x1 = rect.right();
    const F32 y1 = rect.bottom();
    const F32 half = std::numbers::pi_v<F32> * 0.5f;
    const Corner corners[] = {
        {{x0, y0}, {x0 + radius, y0 + radius}, -half, top},
        {{x1, y0}, {x1 - radius, y0 + radius}, 0.0f, top},
        {{x1, y1}, {x1 - radius, y1 - radius}, half, bottom},
        {{x0, y1}, {x0 + radius, y1 - radius}, 2.0f * half, bottom},
    };

    std::vector<Instance> masks;
    masks.reserve(4);
    for (const auto& corner : corners) {
        Instance mask;
        mask.color = corner.color;
        mask.filled = true;
        mask.points.reserve(segments + 2);
        mask.points.push_back(corner.point);
        for (U64 i = 0; i <= segments; ++i) {
            const F32 angle = corner.start - half * static_cast<F32>(i) / static_cast<F32>(segments);
            mask.points.push_back({corner.center.x + std::cos(angle) * radius,
                                   corner.center.y + std::sin(angle) * radius});
        }
        masks.push_back(std::move(mask));
    }
    return masks;
}

struct Polyline::Impl : public Drawable {
    Config config;
    Geometry geometry;
    U64 vertexCapacity = 0;
    bool geometryDirty = true;

    Render::Components::Mesh* mesh() const {
        return as<Render::Components::Mesh>();
    }

    void tessellate() {
        geometry.clear();
        const U64 vertexLimit = std::max<U64>(1, config.capacity);
        const U64 indexLimit = IndexCapacity(vertexLimit);
        const F32 scale = std::max(1e-6f, config.scale);
        const F32 fringe = config.fringe / scale;
        for (const auto& instance : config.instances) {
            if (!instance.visible) {
                continue;
            }
            const U64 vertices = geometry.positions.size();
            const U64 indices = geometry.indices.size();
            if (instance.filled) {
                geometry.fill(instance, fringe);
            } else {
                geometry.stroke(instance, fringe, 1.0f / scale);
            }
            if (geometry.positions.size() > vertexLimit || geometry.indices.size() > indexLimit) {
                geometry.positions.resize(vertices);
                geometry.colors.resize(vertices);
                geometry.indices.resize(indices);
                break;
            }
        }
    }

    Result attach(Context* context, Render::Surface::Config& surfaceConfig) override {
        this->context = context;
        vertexCapacity = std::max<U64>(1, config.capacity);

        Render::Components::Mesh::Config meshConfig;
        meshConfig.vertexCapacity = vertexCapacity;
        meshConfig.indexCapacity = IndexCapacity(vertexCapacity);
        meshConfig.framebufferSize = {
            static_cast<F32>(context->framebufferSize.x),
            static_cast<F32>(context->framebufferSize.y),
        };

        std::shared_ptr<Render::Components::Mesh> created;
        JST_CHECK(context->render->build(created, meshConfig));
        geometryDirty = true;
        return bind(context, std::move(created), surfaceConfig);
    }

    void conceal() override {
        if (component) {
            mesh()->enabled(false);
        }
    }

    Result upload() override {
        if (!component || !context) {
            return Result::SUCCESS;
        }
        auto* mesh = this->mesh();
        mesh->enabled(true);

        JST_CHECK(mesh->updateTransform({
            static_cast<F32>(context->framebufferSize.x),
            static_cast<F32>(context->framebufferSize.y),
        }, {config.origin.x + translation.x, config.origin.y + translation.y}, config.scale));

        JST_CHECK(mesh->updateScissorRect(scissor(config.clip)));

        if (!geometryDirty) {
            return Result::SUCCESS;
        }
        geometryDirty = false;

        const U64 vertexCount = std::min<U64>(geometry.positions.size(), mesh->getConfig().vertexCapacity);
        const U64 indexCount = std::min<U64>(geometry.indices.size(), mesh->getConfig().indexCapacity);

        auto positions = mesh->positions();
        auto colors = mesh->colors();
        auto indices = mesh->indices();
        std::copy_n(geometry.positions.begin(), vertexCount, positions.begin());
        std::copy_n(geometry.colors.begin(), vertexCount, colors.begin());
        U64 validIndices = 0;
        for (U64 i = 0; i + 2 < indexCount; i += 3) {
            if (geometry.indices[i] >= vertexCount ||
                geometry.indices[i + 1] >= vertexCount ||
                geometry.indices[i + 2] >= vertexCount) {
                break;
            }
            indices[i + 0] = geometry.indices[i + 0];
            indices[i + 1] = geometry.indices[i + 1];
            indices[i + 2] = geometry.indices[i + 2];
            validIndices = i + 3;
        }

        JST_CHECK(mesh->update(vertexCount, validIndices));

        return Result::SUCCESS;
    }
};

Polyline::Polyline() {
    this->impl = std::make_unique<Impl>();
}

Polyline::~Polyline() = default;

const Polyline::Config& Polyline::config() const {
    return impl->config;
}

void Polyline::update(Config config) {
    markConfigured();
    const bool geometryChanged = impl->config.instances != config.instances ||
                                 impl->config.fringe != config.fringe ||
                                 impl->config.scale != config.scale ||
                                 impl->config.capacity != config.capacity;
    const bool clipChanged = impl->config.clip != config.clip || impl->config.origin != config.origin;
    const bool resized = impl->attached() && std::max<U64>(1, config.capacity) != impl->vertexCapacity;
    impl->config = std::move(config);
    if (geometryChanged) {
        impl->tessellate();
        impl->geometryDirty = true;
    }
    if (resized) {
        invalidate(Dirty::Resource);
    } else if (geometryChanged || clipChanged) {
        invalidate(Dirty::Paint);
    }
}

Result Polyline::build(Context& ctx) {
    return ctx.attach(*this, this->impl.get());
}

Result Polyline::paint() {
    this->impl->translation = translation();
    this->impl->inherited = inheritedClip();
    return this->impl->paint();
}

void Polyline::conceal() {
    this->impl->conceal();
}

}  // namespace Jetstream::Sakura
