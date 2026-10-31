#include "geomap_tiles.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

#include <zlib.h>

#include <glm/glm.hpp>

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

namespace Jetstream::Render::Components {

namespace {

using Clock = std::chrono::steady_clock;

constexpr F64 Pi = 3.14159265358979323846;
constexpr F64 MaxLatitude = 85.05112878;
constexpr F64 HalfFovTan = 0.414213562373;
constexpr F64 BaseHeight = 1.0 / 0.382683432365 - 1.0;
constexpr U64 MaxNameBytes = 40;
constexpr U64 MaxInflatedBytes = 64ull << 20;
constexpr U64 MaxDownloadBytes = 16ull << 20;
constexpr U64 MaxGeometryPoints = 1u << 22;
constexpr U64 MaxGeometryValues = 3 * MaxGeometryPoints;
constexpr U32 SelectionSamples = 9;
constexpr U32 FootprintCells = 8;
constexpr F32 MaxFillEdge = 1.0f / FootprintCells;
constexpr U8 MaxFallbackLevels = 6;
constexpr U8 PreviewLevels = 2;
constexpr auto AssemblyInterval = std::chrono::milliseconds(250);
constexpr auto DiskMaxAge = std::chrono::hours(24 * 30);

F64 LatitudeToY(F64 lat) {
    const F64 r = std::clamp(lat, -MaxLatitude, MaxLatitude) * Pi / 180.0;
    return (1.0 - std::log(std::tan(r) + 1.0 / std::cos(r)) / Pi) / 2.0;
}

F64 YToLatitude(F64 y) {
    return std::atan(std::sinh(Pi * (1.0 - 2.0 * y))) * 180.0 / Pi;
}

Platform::Http::Client::Response Download(Platform::Http::Client::Request& request, std::string& body) {
    request.onData = [&](std::string_view chunk) {
        if (body.size() + chunk.size() > MaxDownloadBytes) return false;
        body.append(chunk);
        return true;
    };
    return Platform::Http::Client::Send(request);
}

bool Inflate(std::string& data) {
    if (data.size() < 2 || static_cast<U8>(data[0]) != 0x1f ||
        static_cast<U8>(data[1]) != 0x8b) {
        return data.size() <= MaxInflatedBytes;
    }

    z_stream stream{};
    if (inflateInit2(&stream, 15 + 32) != Z_OK) {
        return false;
    }
    std::string output(std::min<U64>(std::max<U64>(data.size() * 3, 4096), MaxInflatedBytes), '\0');
    stream.next_in = reinterpret_cast<Bytef*>(data.data());
    stream.avail_in = static_cast<uInt>(data.size());
    U64 produced = 0;
    int status = Z_OK;
    while (status == Z_OK) {
        if (produced == output.size()) {
            if (output.size() >= MaxInflatedBytes) break;
            output.resize(std::min<U64>(output.size() * 2, MaxInflatedBytes));
        }
        stream.next_out = reinterpret_cast<Bytef*>(output.data() + produced);
        stream.avail_out = static_cast<uInt>(output.size() - produced);
        status = inflate(&stream, Z_NO_FLUSH);
        produced = output.size() - stream.avail_out;
    }
    inflateEnd(&stream);
    if (status != Z_STREAM_END) {
        return false;
    }
    output.resize(produced);
    data = std::move(output);
    return true;
}

bool RenderableName(std::string_view name) {
    if (name.empty() || name.size() > MaxNameBytes) return false;
    for (U64 i = 0; i < name.size();) {
        const U8 lead = static_cast<U8>(name[i]);
        U32 codepoint = 0;
        if (lead < 0x80) {
            codepoint = lead;
            i += 1;
        } else if ((lead & 0xE0) == 0xC0 && i + 1 < name.size() &&
                   (static_cast<U8>(name[i + 1]) & 0xC0) == 0x80) {
            codepoint = ((lead & 0x1F) << 6) | (static_cast<U8>(name[i + 1]) & 0x3F);
            i += 2;
        } else {
            return false;
        }
        const bool ascii = codepoint >= 0x20 && codepoint <= 0x7E;
        const bool latin = codepoint >= 0xA0 && codepoint <= 0x17F;
        if (!ascii && !latin) return false;
    }
    return true;
}

struct Rect {
    F64 x0 = 0.0;
    F64 y0 = 0.0;
    F64 x1 = 1.0;
    F64 y1 = 1.0;

    bool contains(F64 x, F64 y) const {
        return x >= x0 && x < x1 && y >= y0 && y < y1;
    }
};

bool ClipSegment(F64& x0, F64& y0, F64& x1, F64& y1, const Rect& rect) {
    const F64 dx = x1 - x0;
    const F64 dy = y1 - y0;
    F64 t0 = 0.0;
    F64 t1 = 1.0;
    auto edge = [&](F64 p, F64 q) {
        if (p == 0.0) return q >= 0.0;
        const F64 t = q / p;
        if (p < 0.0) {
            if (t > t1) return false;
            t0 = std::max(t0, t);
        } else {
            if (t < t0) return false;
            t1 = std::min(t1, t);
        }
        return true;
    };
    if (!edge(-dx, x0 - rect.x0) || !edge(dx, rect.x1 - x0) ||
        !edge(-dy, y0 - rect.y0) || !edge(dy, rect.y1 - y0) || t1 <= t0) {
        return false;
    }
    const F64 sx = x0;
    const F64 sy = y0;
    x0 = sx + t0 * dx;
    y0 = sy + t0 * dy;
    x1 = sx + t1 * dx;
    y1 = sy + t1 * dy;
    return true;
}

template <typename Positions, typename Emit>
void ClipTriangles(const Positions& positions,
                   const std::vector<U32>& triangles,
                   const Rect& rect,
                   Emit&& emit,
                   std::vector<U32>& indices) {
    std::vector<U32> remap(positions.size() / 2, std::numeric_limits<U32>::max());
    auto shared = [&](U32 vertex) {
        auto& slot = remap[vertex];
        if (slot == std::numeric_limits<U32>::max()) {
            slot = emit(positions[vertex * 2], positions[vertex * 2 + 1]);
        }
        return slot;
    };

    struct Point { F64 x, y; };
    std::array<Point, 16> polygon{};
    std::array<Point, 16> scratch{};

    for (U64 t = 0; t + 2 < triangles.size(); t += 3) {
        const U32 a = triangles[t];
        const U32 b = triangles[t + 1];
        const U32 c = triangles[t + 2];
        const Point pa{positions[a * 2], positions[a * 2 + 1]};
        const Point pb{positions[b * 2], positions[b * 2 + 1]};
        const Point pc{positions[c * 2], positions[c * 2 + 1]};
        const F64 minX = std::min({pa.x, pb.x, pc.x});
        const F64 maxX = std::max({pa.x, pb.x, pc.x});
        const F64 minY = std::min({pa.y, pb.y, pc.y});
        const F64 maxY = std::max({pa.y, pb.y, pc.y});
        if (maxX <= rect.x0 || minX >= rect.x1 || maxY <= rect.y0 || minY >= rect.y1) {
            continue;
        }
        if (minX >= rect.x0 && maxX <= rect.x1 && minY >= rect.y0 && maxY <= rect.y1) {
            indices.push_back(shared(a));
            indices.push_back(shared(b));
            indices.push_back(shared(c));
            continue;
        }

        U64 count = 3;
        polygon[0] = pa;
        polygon[1] = pb;
        polygon[2] = pc;
        for (U32 side = 0; side < 4 && count > 0; ++side) {
            auto inside = [&](const Point& p) {
                switch (side) {
                    case 0: return p.x >= rect.x0;
                    case 1: return p.x <= rect.x1;
                    case 2: return p.y >= rect.y0;
                    default: return p.y <= rect.y1;
                }
            };
            auto cross = [&](const Point& p, const Point& q) {
                F64 t = 0.0;
                switch (side) {
                    case 0: t = (rect.x0 - p.x) / (q.x - p.x); break;
                    case 1: t = (rect.x1 - p.x) / (q.x - p.x); break;
                    case 2: t = (rect.y0 - p.y) / (q.y - p.y); break;
                    default: t = (rect.y1 - p.y) / (q.y - p.y); break;
                }
                return Point{p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t};
            };
            U64 next = 0;
            for (U64 i = 0; i < count; ++i) {
                const Point& current = polygon[i];
                const Point& previous = polygon[(i + count - 1) % count];
                const bool currentInside = inside(current);
                if (currentInside != inside(previous)) {
                    scratch[next++] = cross(previous, current);
                }
                if (currentInside) {
                    scratch[next++] = current;
                }
            }
            count = next;
            std::copy_n(scratch.begin(), count, polygon.begin());
        }
        if (count < 3) continue;
        const U32 first = emit(polygon[0].x, polygon[0].y);
        U32 previous = emit(polygon[1].x, polygon[1].y);
        for (U64 i = 2; i < count; ++i) {
            const U32 current = emit(polygon[i].x, polygon[i].y);
            indices.push_back(first);
            indices.push_back(previous);
            indices.push_back(current);
            previous = current;
        }
    }
}

class Earcut {
 public:
    void run(const std::vector<F64>& coords,
             const std::vector<U32>& holes,
             std::vector<U32>& triangles) {
        nodes.clear();
        output = &triangles;
        const U32 count = static_cast<U32>(coords.size() / 2);
        const U32 outerEnd = holes.empty() ? count : holes.front();
        Node* outer = linkedList(coords, 0, outerEnd, true);
        if (!outer || outer->next == outer->prev) return;
        if (!holes.empty()) outer = eliminateHoles(coords, holes, outer);

        invSize = 0.0;
        if (count > 80) {
            minX = coords[0];
            minY = coords[1];
            F64 maxX = minX;
            F64 maxY = minY;
            for (U32 i = 1; i < outerEnd; ++i) {
                minX = std::min(minX, coords[i * 2]);
                minY = std::min(minY, coords[i * 2 + 1]);
                maxX = std::max(maxX, coords[i * 2]);
                maxY = std::max(maxY, coords[i * 2 + 1]);
            }
            const F64 size = std::max(maxX - minX, maxY - minY);
            invSize = size != 0.0 ? 32767.0 / size : 0.0;
        }
        earcutLinked(outer, 0);
    }

 private:
    struct Node {
        U32 i = 0;
        F64 x = 0.0;
        F64 y = 0.0;
        Node* prev = nullptr;
        Node* next = nullptr;
        I32 z = 0;
        Node* prevZ = nullptr;
        Node* nextZ = nullptr;
        bool steiner = false;
    };

    std::deque<Node> nodes;
    std::vector<U32>* output = nullptr;
    F64 minX = 0.0;
    F64 minY = 0.0;
    F64 invSize = 0.0;

    Node* insertNode(U32 i, F64 x, F64 y, Node* last) {
        Node* p = &nodes.emplace_back();
        p->i = i;
        p->x = x;
        p->y = y;
        if (!last) {
            p->prev = p;
            p->next = p;
        } else {
            p->next = last->next;
            p->prev = last;
            last->next->prev = p;
            last->next = p;
        }
        return p;
    }

    static void removeNode(Node* p) {
        p->next->prev = p->prev;
        p->prev->next = p->next;
        if (p->prevZ) p->prevZ->nextZ = p->nextZ;
        if (p->nextZ) p->nextZ->prevZ = p->prevZ;
    }

    static F64 area(const Node* p, const Node* q, const Node* r) {
        return (q->y - p->y) * (r->x - q->x) - (q->x - p->x) * (r->y - q->y);
    }

    static bool equals(const Node* a, const Node* b) {
        return a->x == b->x && a->y == b->y;
    }

    static bool pointInTriangle(F64 ax, F64 ay, F64 bx, F64 by, F64 cx, F64 cy,
                                F64 px, F64 py) {
        return (cx - px) * (ay - py) >= (ax - px) * (cy - py) &&
               (ax - px) * (by - py) >= (bx - px) * (ay - py) &&
               (bx - px) * (cy - py) >= (cx - px) * (by - py);
    }

    static I32 sign(F64 value) {
        return value > 0.0 ? 1 : value < 0.0 ? -1 : 0;
    }

    static bool onSegment(const Node* p, const Node* q, const Node* r) {
        return q->x <= std::max(p->x, r->x) && q->x >= std::min(p->x, r->x) &&
               q->y <= std::max(p->y, r->y) && q->y >= std::min(p->y, r->y);
    }

    static bool intersects(const Node* p1, const Node* q1, const Node* p2, const Node* q2) {
        const I32 o1 = sign(area(p1, q1, p2));
        const I32 o2 = sign(area(p1, q1, q2));
        const I32 o3 = sign(area(p2, q2, p1));
        const I32 o4 = sign(area(p2, q2, q1));
        if (o1 != o2 && o3 != o4) return true;
        if (o1 == 0 && onSegment(p1, p2, q1)) return true;
        if (o2 == 0 && onSegment(p1, q2, q1)) return true;
        if (o3 == 0 && onSegment(p2, p1, q2)) return true;
        if (o4 == 0 && onSegment(p2, q1, q2)) return true;
        return false;
    }

    static bool intersectsPolygon(const Node* a, const Node* b) {
        const Node* p = a;
        do {
            if (p->i != a->i && p->next->i != a->i && p->i != b->i && p->next->i != b->i &&
                intersects(p, p->next, a, b)) {
                return true;
            }
            p = p->next;
        } while (p != a);
        return false;
    }

    static bool locallyInside(const Node* a, const Node* b) {
        return area(a->prev, a, a->next) < 0.0
            ? area(a, b, a->next) >= 0.0 && area(a, a->prev, b) >= 0.0
            : area(a, b, a->prev) < 0.0 || area(a, a->next, b) < 0.0;
    }

    static bool middleInside(const Node* a, const Node* b) {
        const Node* p = a;
        bool inside = false;
        const F64 px = (a->x + b->x) / 2.0;
        const F64 py = (a->y + b->y) / 2.0;
        do {
            if (((p->y > py) != (p->next->y > py)) && p->next->y != p->y &&
                (px < (p->next->x - p->x) * (py - p->y) / (p->next->y - p->y) + p->x)) {
                inside = !inside;
            }
            p = p->next;
        } while (p != a);
        return inside;
    }

    static bool isValidDiagonal(const Node* a, const Node* b) {
        return a->next->i != b->i && a->prev->i != b->i && !intersectsPolygon(a, b) &&
               ((locallyInside(a, b) && locallyInside(b, a) && middleInside(a, b) &&
                 (area(a->prev, a, b->prev) != 0.0 || area(a, b->prev, b) != 0.0)) ||
                (equals(a, b) && area(a->prev, a, a->next) > 0.0 &&
                 area(b->prev, b, b->next) > 0.0));
    }

    Node* splitPolygon(Node* a, Node* b) {
        Node* a2 = &nodes.emplace_back(Node{a->i, a->x, a->y});
        Node* b2 = &nodes.emplace_back(Node{b->i, b->x, b->y});
        Node* an = a->next;
        Node* bp = b->prev;
        a->next = b;
        b->prev = a;
        a2->next = an;
        an->prev = a2;
        b2->next = a2;
        a2->prev = b2;
        bp->next = b2;
        b2->prev = bp;
        return b2;
    }

    Node* linkedList(const std::vector<F64>& coords, U32 start, U32 end, bool clockwise) {
        F64 sum = 0.0;
        for (U32 i = start, j = end - 1; i < end; j = i++) {
            sum += (coords[j * 2] - coords[i * 2]) * (coords[i * 2 + 1] + coords[j * 2 + 1]);
        }
        Node* last = nullptr;
        if (clockwise == (sum > 0.0)) {
            for (U32 i = start; i < end; ++i) {
                last = insertNode(i, coords[i * 2], coords[i * 2 + 1], last);
            }
        } else {
            for (U32 i = end; i-- > start;) {
                last = insertNode(i, coords[i * 2], coords[i * 2 + 1], last);
            }
        }
        if (last && equals(last, last->next)) {
            removeNode(last);
            last = last->next;
        }
        return last;
    }

    static Node* filterPoints(Node* start, Node* end = nullptr) {
        if (!start) return start;
        if (!end) end = start;
        Node* p = start;
        bool again = false;
        do {
            again = false;
            if (!p->steiner && (equals(p, p->next) || area(p->prev, p, p->next) == 0.0)) {
                removeNode(p);
                p = end = p->prev;
                if (p == p->next) break;
                again = true;
            } else {
                p = p->next;
            }
        } while (again || p != end);
        return end;
    }

    I32 zOrder(F64 px, F64 py) const {
        U32 x = static_cast<U32>((px - minX) * invSize);
        U32 y = static_cast<U32>((py - minY) * invSize);
        x = (x | (x << 8)) & 0x00FF00FF;
        x = (x | (x << 4)) & 0x0F0F0F0F;
        x = (x | (x << 2)) & 0x33333333;
        x = (x | (x << 1)) & 0x55555555;
        y = (y | (y << 8)) & 0x00FF00FF;
        y = (y | (y << 4)) & 0x0F0F0F0F;
        y = (y | (y << 2)) & 0x33333333;
        y = (y | (y << 1)) & 0x55555555;
        return static_cast<I32>(x | (y << 1));
    }

    static Node* sortLinked(Node* list) {
        U32 inSize = 1;
        U32 numMerges = 0;
        do {
            Node* p = list;
            list = nullptr;
            Node* tail = nullptr;
            numMerges = 0;
            while (p) {
                ++numMerges;
                Node* q = p;
                U32 pSize = 0;
                for (U32 i = 0; i < inSize; ++i) {
                    ++pSize;
                    q = q->nextZ;
                    if (!q) break;
                }
                U32 qSize = inSize;
                while (pSize > 0 || (qSize > 0 && q)) {
                    Node* e = nullptr;
                    if (pSize != 0 && (qSize == 0 || !q || p->z <= q->z)) {
                        e = p;
                        p = p->nextZ;
                        --pSize;
                    } else {
                        e = q;
                        q = q->nextZ;
                        --qSize;
                    }
                    if (tail) {
                        tail->nextZ = e;
                    } else {
                        list = e;
                    }
                    e->prevZ = tail;
                    tail = e;
                }
                p = q;
            }
            tail->nextZ = nullptr;
            inSize *= 2;
        } while (numMerges > 1);
        return list;
    }

    void indexCurve(Node* start) {
        Node* p = start;
        do {
            if (p->z == 0) p->z = zOrder(p->x, p->y);
            p->prevZ = p->prev;
            p->nextZ = p->next;
            p = p->next;
        } while (p != start);
        p->prevZ->nextZ = nullptr;
        p->prevZ = nullptr;
        sortLinked(p);
    }

    static bool isEar(const Node* ear) {
        const Node* a = ear->prev;
        const Node* b = ear;
        const Node* c = ear->next;
        if (area(a, b, c) >= 0.0) return false;
        const F64 x0 = std::min({a->x, b->x, c->x});
        const F64 y0 = std::min({a->y, b->y, c->y});
        const F64 x1 = std::max({a->x, b->x, c->x});
        const F64 y1 = std::max({a->y, b->y, c->y});
        const Node* p = c->next;
        while (p != a) {
            if (p->x >= x0 && p->x <= x1 && p->y >= y0 && p->y <= y1 &&
                pointInTriangle(a->x, a->y, b->x, b->y, c->x, c->y, p->x, p->y) &&
                area(p->prev, p, p->next) >= 0.0) {
                return false;
            }
            p = p->next;
        }
        return true;
    }

    bool isEarHashed(const Node* ear) const {
        const Node* a = ear->prev;
        const Node* b = ear;
        const Node* c = ear->next;
        if (area(a, b, c) >= 0.0) return false;
        const F64 x0 = std::min({a->x, b->x, c->x});
        const F64 y0 = std::min({a->y, b->y, c->y});
        const F64 x1 = std::max({a->x, b->x, c->x});
        const F64 y1 = std::max({a->y, b->y, c->y});
        const I32 minZ = zOrder(x0, y0);
        const I32 maxZ = zOrder(x1, y1);
        auto blocks = [&](const Node* p) {
            return p->x >= x0 && p->x <= x1 && p->y >= y0 && p->y <= y1 && p != a && p != c &&
                   pointInTriangle(a->x, a->y, b->x, b->y, c->x, c->y, p->x, p->y) &&
                   area(p->prev, p, p->next) >= 0.0;
        };
        const Node* p = ear->prevZ;
        const Node* n = ear->nextZ;
        while (p && p->z >= minZ && n && n->z <= maxZ) {
            if (blocks(p)) return false;
            p = p->prevZ;
            if (blocks(n)) return false;
            n = n->nextZ;
        }
        while (p && p->z >= minZ) {
            if (blocks(p)) return false;
            p = p->prevZ;
        }
        while (n && n->z <= maxZ) {
            if (blocks(n)) return false;
            n = n->nextZ;
        }
        return true;
    }

    void emit(const Node* a, const Node* b, const Node* c) {
        output->push_back(a->i);
        output->push_back(b->i);
        output->push_back(c->i);
    }

    Node* cureLocalIntersections(Node* start) {
        Node* p = start;
        do {
            Node* a = p->prev;
            Node* b = p->next->next;
            if (!equals(a, b) && intersects(a, p, p->next, b) &&
                locallyInside(a, b) && locallyInside(b, a)) {
                emit(a, p, b);
                removeNode(p);
                removeNode(p->next);
                p = start = b;
            }
            p = p->next;
        } while (p != start);
        return filterPoints(p);
    }

    void splitEarcut(Node* start) {
        Node* a = start;
        do {
            Node* b = a->next->next;
            while (b != a->prev) {
                if (a->i != b->i && isValidDiagonal(a, b)) {
                    Node* c = splitPolygon(a, b);
                    a = filterPoints(a, a->next);
                    c = filterPoints(c, c->next);
                    earcutLinked(a, 0);
                    earcutLinked(c, 0);
                    return;
                }
                b = b->next;
            }
            a = a->next;
        } while (a != start);
    }

    void earcutLinked(Node* ear, U32 pass) {
        if (!ear) return;
        if (pass == 0 && invSize != 0.0) indexCurve(ear);
        Node* stop = ear;
        while (ear->prev != ear->next) {
            Node* prev = ear->prev;
            Node* next = ear->next;
            if (invSize != 0.0 ? isEarHashed(ear) : isEar(ear)) {
                emit(prev, ear, next);
                removeNode(ear);
                ear = next->next;
                stop = next->next;
                continue;
            }
            ear = next;
            if (ear == stop) {
                if (pass == 0) {
                    earcutLinked(filterPoints(ear), 1);
                } else if (pass == 1) {
                    earcutLinked(cureLocalIntersections(filterPoints(ear)), 2);
                } else {
                    splitEarcut(ear);
                }
                break;
            }
        }
    }

    static Node* leftmost(Node* start) {
        Node* p = start;
        Node* result = start;
        do {
            if (p->x < result->x || (p->x == result->x && p->y < result->y)) result = p;
            p = p->next;
        } while (p != start);
        return result;
    }

    static bool sectorContainsSector(const Node* m, const Node* p) {
        return area(m->prev, m, p->prev) < 0.0 && area(p->next, m, m->next) < 0.0;
    }

    static Node* findHoleBridge(Node* hole, Node* outer) {
        Node* p = outer;
        const F64 hx = hole->x;
        const F64 hy = hole->y;
        F64 qx = -std::numeric_limits<F64>::infinity();
        Node* m = nullptr;
        do {
            if (hy <= p->y && hy >= p->next->y && p->next->y != p->y) {
                const F64 x = p->x + (hy - p->y) * (p->next->x - p->x) / (p->next->y - p->y);
                if (x <= hx && x > qx) {
                    qx = x;
                    m = p->x < p->next->x ? p : p->next;
                    if (x == hx) return m;
                }
            }
            p = p->next;
        } while (p != outer);
        if (!m) return nullptr;

        Node* stop = m;
        const F64 mx = m->x;
        const F64 my = m->y;
        F64 tanMin = std::numeric_limits<F64>::infinity();
        p = m;
        do {
            if (hx >= p->x && p->x >= mx && hx != p->x &&
                pointInTriangle(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p->x, p->y)) {
                const F64 tan = std::abs(hy - p->y) / (hx - p->x);
                if (locallyInside(p, hole) &&
                    (tan < tanMin || (tan == tanMin &&
                     (p->x > m->x || (p->x == m->x && sectorContainsSector(m, p)))))) {
                    m = p;
                    tanMin = tan;
                }
            }
            p = p->next;
        } while (p != stop);
        return m;
    }

    Node* eliminateHoles(const std::vector<F64>& coords, const std::vector<U32>& holes, Node* outer) {
        std::vector<Node*> queue;
        const U32 count = static_cast<U32>(coords.size() / 2);
        for (U64 i = 0; i < holes.size(); ++i) {
            const U32 start = holes[i];
            const U32 end = i + 1 < holes.size() ? holes[i + 1] : count;
            Node* list = linkedList(coords, start, end, false);
            if (!list) continue;
            if (list == list->next) list->steiner = true;
            queue.push_back(leftmost(list));
        }
        std::sort(queue.begin(), queue.end(), [](const Node* a, const Node* b) {
            return a->x < b->x;
        });
        for (Node* hole : queue) {
            Node* bridge = findHoleBridge(hole, outer);
            if (!bridge) continue;
            Node* reverse = splitPolygon(bridge, hole);
            filterPoints(reverse, reverse->next);
            outer = filterPoints(bridge, bridge->next);
        }
        return outer;
    }
};

class Proto {
 public:
    explicit Proto(std::string_view data)
        : cursor(reinterpret_cast<const U8*>(data.data())),
          end(reinterpret_cast<const U8*>(data.data()) + data.size()) {}

    bool next() {
        if (cursor >= end) return false;
        U64 key = 0;
        if (!varint(key)) return false;
        field = static_cast<U32>(key >> 3);
        wire = static_cast<U32>(key & 7);
        return true;
    }

    bool varint(U64& value) {
        value = 0;
        for (U32 shift = 0; shift < 64; shift += 7) {
            if (cursor >= end) return fail();
            const U8 byte = *cursor++;
            if (shift == 63 && (byte & 0xFE)) return fail();
            value |= static_cast<U64>(byte & 0x7F) << shift;
            if (!(byte & 0x80)) return true;
        }
        return fail();
    }

    bool bytes(std::string_view& value) {
        U64 size = 0;
        if (!varint(size)) return false;
        if (size > static_cast<U64>(end - cursor)) return fail();
        value = {reinterpret_cast<const char*>(cursor), static_cast<size_t>(size)};
        cursor += size;
        return true;
    }

    bool fixed32(U32& value) {
        if (end - cursor < 4) return fail();
        std::memcpy(&value, cursor, 4);
        cursor += 4;
        return true;
    }

    bool fixed64(U64& value) {
        if (end - cursor < 8) return fail();
        std::memcpy(&value, cursor, 8);
        cursor += 8;
        return true;
    }

    bool skip() {
        U64 ignored = 0;
        U32 ignored32 = 0;
        std::string_view view;
        switch (wire) {
            case 0: return varint(ignored);
            case 1: return fixed64(ignored);
            case 2: return bytes(view);
            case 5: return fixed32(ignored32);
            default: return fail();
        }
    }

    bool good() const {
        return valid;
    }

    bool repeated(std::vector<U32>& values, U64 limit = MaxInflatedBytes / sizeof(U32)) {
        auto append = [&](U64 value) {
            if (value > UINT32_MAX || values.size() >= limit) return fail();
            values.push_back(static_cast<U32>(value));
            return true;
        };
        U64 value = 0;
        if (wire == 0) {
            return varint(value) && append(value);
        }
        if (wire != 2) return fail();
        std::string_view view;
        if (!bytes(view)) return false;
        Proto inner(view);
        while (!inner.done()) {
            if (!inner.varint(value) || !append(value)) return fail();
        }
        return true;
    }

    bool done() const {
        return cursor >= end;
    }

    U32 field = 0;
    U32 wire = 0;

 private:
    bool fail() {
        valid = false;
        cursor = end;
        return false;
    }

    const U8* cursor;
    const U8* end;
    bool valid = true;
};

}  // namespace

struct TileData {
    std::array<std::vector<F32>, GeoMapTiles::LineCount> lines;
    std::array<GeoMapTiles::Mesh, GeoMapTiles::FillCount> fills;
    std::vector<GeoMapTiles::Label> labels;
    bool missing = false;

    U64 bytes() const {
        U64 total = sizeof(TileData);
        for (const auto& line : lines) total += line.size() * sizeof(F32);
        for (const auto& fill : fills) {
            total += fill.positions.size() * sizeof(F32) + fill.indices.size() * sizeof(U32);
        }
        for (const auto& label : labels) total += sizeof(label) + label.name.size();
        return total;
    }
};

namespace {

void Subdivide(GeoMapTiles::Mesh& mesh) {
    constexpr F32 limit = MaxFillEdge * MaxFillEdge;
    auto& positions = mesh.positions;
    std::unordered_map<U64, U32> midpoints;
    auto length = [&](U32 a, U32 b) {
        const F32 dx = positions[a * 2] - positions[b * 2];
        const F32 dy = positions[a * 2 + 1] - positions[b * 2 + 1];
        return dx * dx + dy * dy;
    };
    auto midpoint = [&](U32 a, U32 b) {
        const U64 key = a < b ? (static_cast<U64>(a) << 32 | b) : (static_cast<U64>(b) << 32 | a);
        const auto [it, inserted] = midpoints.try_emplace(key, 0);
        if (inserted) {
            it->second = static_cast<U32>(positions.size() / 2);
            const F32 x = (positions[a * 2] + positions[b * 2]) * 0.5f;
            const F32 y = (positions[a * 2 + 1] + positions[b * 2 + 1]) * 0.5f;
            positions.push_back(x);
            positions.push_back(y);
        }
        return it->second;
    };

    std::vector<U32> out;
    out.reserve(mesh.indices.size());
    std::vector<std::array<U32, 3>> pending;
    for (U64 i = 0; i + 2 < mesh.indices.size(); i += 3) {
        pending.push_back({mesh.indices[i], mesh.indices[i + 1], mesh.indices[i + 2]});
        while (!pending.empty()) {
            const auto [a, b, c] = pending.back();
            pending.pop_back();
            const F32 ab = length(a, b);
            const F32 bc = length(b, c);
            const F32 ca = length(c, a);
            if (std::max({ab, bc, ca}) <= limit) {
                out.insert(out.end(), {a, b, c});
                continue;
            }
            if (ab >= bc && ab >= ca) {
                const U32 m = midpoint(a, b);
                pending.push_back({a, m, c});
                pending.push_back({m, b, c});
            } else if (bc >= ca) {
                const U32 m = midpoint(b, c);
                pending.push_back({a, b, m});
                pending.push_back({a, m, c});
            } else {
                const U32 m = midpoint(c, a);
                pending.push_back({a, b, m});
                pending.push_back({m, b, c});
            }
        }
    }
    mesh.indices = std::move(out);
}

class Decoder {
 public:
    std::shared_ptr<TileData> run(std::string_view data) {
        tile = std::make_shared<TileData>();
        Proto message(data);
        while (message.next()) {
            if (message.field == 3 && message.wire == 2) {
                std::string_view layer;
                if (!message.bytes(layer) || !decodeLayer(layer)) return nullptr;
            } else if (!message.skip()) {
                return nullptr;
            }
        }
        if (!message.good()) return nullptr;
        for (auto& mesh : tile->fills) Subdivide(mesh);
        return std::move(tile);
    }

 private:
    using Line = GeoMapTiles::Line;
    using Fill = GeoMapTiles::Fill;
    using Place = GeoMapTiles::Place;
    using Point = std::array<I32, 2>;
    using Ring = std::vector<Point>;

    struct Value {
        std::string_view text;
        F64 number = 0.0;
        bool isText = false;
    };

    struct Feature {
        U32 type = 0;
        std::vector<U32> tags;
        std::vector<U32> geometry;
    };

    struct Layer {
        std::string_view name;
        F64 extent = 4096.0;
        std::vector<std::string_view> keys;
        std::vector<Value> values;
        std::vector<std::string_view> features;

        std::optional<U32> key(std::string_view name) const {
            for (U32 i = 0; i < keys.size(); ++i) {
                if (keys[i] == name) return i;
            }
            return std::nullopt;
        }
    };

    std::shared_ptr<TileData> tile;
    std::vector<Ring> parts;
    std::vector<std::array<I32, 4>> oceanEdges;
    std::vector<std::array<I32, 4>> waterEdges;
    Earcut earcut;
    std::vector<F64> coords;
    std::vector<U32> holes;
    std::vector<U32> triangles;

    static bool decodeValue(std::string_view data, Value& value) {
        Proto message(data);
        while (message.next()) {
            U64 raw = 0;
            U32 raw32 = 0;
            switch (message.field) {
                case 1:
                    if (!message.bytes(value.text)) return false;
                    value.isText = true;
                    break;
                case 2: {
                    if (!message.fixed32(raw32)) return false;
                    F32 number = 0.0f;
                    std::memcpy(&number, &raw32, 4);
                    value.number = number;
                    break;
                }
                case 3: {
                    if (!message.fixed64(raw)) return false;
                    std::memcpy(&value.number, &raw, 8);
                    break;
                }
                case 4:
                    if (!message.varint(raw)) return false;
                    value.number = static_cast<F64>(static_cast<I64>(raw));
                    break;
                case 5:
                case 7:
                    if (!message.varint(raw)) return false;
                    value.number = static_cast<F64>(raw);
                    break;
                case 6:
                    if (!message.varint(raw)) return false;
                    value.number = static_cast<F64>(static_cast<I64>(raw >> 1) ^ -static_cast<I64>(raw & 1));
                    break;
                default:
                    if (!message.skip()) return false;
                    break;
            }
        }
        return message.good();
    }

    static bool decodeFeature(std::string_view data, Feature& feature) {
        Proto message(data);
        while (message.next()) {
            U64 raw = 0;
            if (message.field == 2 && (message.wire == 0 || message.wire == 2)) {
                if (!message.repeated(feature.tags)) return false;
            } else if (message.field == 3 && message.wire == 0) {
                if (!message.varint(raw)) return false;
                feature.type = static_cast<U32>(raw);
            } else if (message.field == 4 && (message.wire == 0 || message.wire == 2)) {
                if (!message.repeated(feature.geometry, MaxGeometryValues)) return false;
            } else if (!message.skip()) {
                return false;
            }
        }
        return message.good();
    }

    bool decodeGeometry(const std::vector<U32>& data, U32 type) {
        parts.clear();
        if (type < 1 || type > 3) return false;
        U64 cursor = 0;
        I64 x = 0;
        I64 y = 0;
        U64 points = 0;
        U64 nextCommand = 1;
        bool complete = false;
        while (cursor < data.size()) {
            const U32 command = data[cursor++];
            const U64 id = command & 7;
            const U64 count = command >> 3;
            if (id != nextCommand || count == 0) return false;

            // MVT points have one MoveTo; each line has MoveTo + LineTo;
            // each polygon ring has MoveTo + LineTo (at least two) + ClosePath.
            if (id == 1) {
                if (type != 1 && count != 1) return false;
                nextCommand = type == 1 ? 0 : 2;
                complete = type == 1;
            } else if (id == 2) {
                if (type == 3 && count < 2) return false;
                nextCommand = type == 3 ? 7 : 1;
                complete = type == 2;
            } else if (id == 7) {
                if (count != 1) return false;
                nextCommand = 1;
                complete = true;
                continue;
            } else {
                return false;
            }
            if (count > (data.size() - cursor) / 2) return false;
            for (U64 i = 0; i < count; ++i) {
                const U32 dx = data[cursor++];
                const U32 dy = data[cursor++];
                if (id == 2 && dx == 0 && dy == 0) return false;
                x += static_cast<I64>(dx >> 1) ^ -static_cast<I64>(dx & 1);
                y += static_cast<I64>(dy >> 1) ^ -static_cast<I64>(dy & 1);
                if (std::abs(x) > (1 << 20) || std::abs(y) > (1 << 20) || ++points > MaxGeometryPoints) {
                    return false;
                }
                if (id == 1) parts.emplace_back();
                parts.back().push_back({static_cast<I32>(x), static_cast<I32>(y)});
            }
        }
        return complete;
    }

    void addLines(Line line, F64 extent) {
        const Rect rect{0.0, 0.0, extent, extent};
        auto& out = tile->lines[static_cast<U64>(line)];
        for (const auto& part : parts) {
            for (U64 i = 1; i < part.size(); ++i) {
                F64 x0 = part[i - 1][0];
                F64 y0 = part[i - 1][1];
                F64 x1 = part[i][0];
                F64 y1 = part[i][1];
                if (!ClipSegment(x0, y0, x1, y1, rect)) continue;
                out.insert(out.end(), {static_cast<F32>(x0 / extent), static_cast<F32>(y0 / extent),
                                       static_cast<F32>(x1 / extent), static_cast<F32>(y1 / extent)});
            }
        }
    }

    bool addPolygons(Fill fill, F64 extent, std::vector<std::array<I32, 4>>* edges) {
        auto& mesh = tile->fills[static_cast<U64>(fill)];
        const Rect rect{0.0, 0.0, extent, extent};
        auto emit = [&](F64 x, F64 y) {
            mesh.positions.push_back(static_cast<F32>(x / extent));
            mesh.positions.push_back(static_cast<F32>(y / extent));
            return static_cast<U32>(mesh.positions.size() / 2 - 1);
        };
        auto flush = [&]() {
            if (coords.size() >= 6) {
                triangles.clear();
                earcut.run(coords, holes, triangles);
                ClipTriangles(coords, triangles, rect, emit, mesh.indices);
            }
            coords.clear();
            holes.clear();
        };

        coords.clear();
        holes.clear();
        bool open = false;
        for (const auto& ring : parts) {
            if (ring.size() < 3 || ring.front() == ring.back()) return false;
            F64 area = 0.0;
            for (U64 i = 0, j = ring.size() - 1; i < ring.size(); j = i++) {
                area += static_cast<F64>(ring[j][0]) * ring[i][1] -
                        static_cast<F64>(ring[i][0]) * ring[j][1];
            }
            if (area == 0.0) return false;
            if (area > 0.0) {
                flush();
                open = true;
            } else if (!open) {
                return false;
            } else {
                holes.push_back(static_cast<U32>(coords.size() / 2));
            }
            for (const auto& point : ring) {
                coords.push_back(point[0]);
                coords.push_back(point[1]);
            }
            if (edges) {
                for (U64 i = 0; i < ring.size(); ++i) {
                    const auto& a = ring[i];
                    const auto& b = ring[(i + 1) % ring.size()];
                    edges->push_back({a[0], a[1], b[0], b[1]});
                }
            }
        }
        flush();
        return true;
    }

    void addCoastline(F64 extent) {
        auto pack = [](I32 x0, I32 y0, I32 x1, I32 y1) {
            return (static_cast<U64>(static_cast<U16>(x0)) << 48) |
                   (static_cast<U64>(static_cast<U16>(y0)) << 32) |
                   (static_cast<U64>(static_cast<U16>(x1)) << 16) |
                   static_cast<U64>(static_cast<U16>(y1));
        };
        std::unordered_set<U64> edges;
        edges.reserve(oceanEdges.size());
        for (const auto& e : oceanEdges) edges.insert(pack(e[0], e[1], e[2], e[3]));
        std::unordered_set<U64> water;
        water.reserve(waterEdges.size() * 2);
        for (const auto& e : waterEdges) {
            water.insert(pack(e[0], e[1], e[2], e[3]));
            water.insert(pack(e[2], e[3], e[0], e[1]));
        }

        const Rect rect{0.0, 0.0, extent, extent};
        auto& out = tile->lines[static_cast<U64>(Line::Coastline)];
        auto border = [&](F64 a, F64 b) {
            return (a <= 0.5 && b <= 0.5) || (a >= extent - 0.5 && b >= extent - 0.5);
        };
        for (const auto& e : oceanEdges) {
            if (edges.contains(pack(e[2], e[3], e[0], e[1])) ||
                water.contains(pack(e[0], e[1], e[2], e[3]))) {
                continue;
            }
            F64 x0 = e[0];
            F64 y0 = e[1];
            F64 x1 = e[2];
            F64 y1 = e[3];
            if (!ClipSegment(x0, y0, x1, y1, rect)) continue;
            if (border(x0, x1) || border(y0, y1)) continue;
            out.insert(out.end(), {static_cast<F32>(x0 / extent), static_cast<F32>(y0 / extent),
                                   static_cast<F32>(x1 / extent), static_cast<F32>(y1 / extent)});
        }
        oceanEdges.clear();
        waterEdges.clear();
    }

    bool decodeLayer(std::string_view data) {
        Layer layer;
        Proto message(data);
        while (message.next()) {
            U64 raw = 0;
            std::string_view view;
            if (message.field == 1 && message.wire == 2) {
                if (!message.bytes(layer.name)) return false;
            } else if (message.field == 2 && message.wire == 2) {
                if (!message.bytes(view)) return false;
                layer.features.push_back(view);
            } else if (message.field == 3 && message.wire == 2) {
                if (!message.bytes(view)) return false;
                layer.keys.push_back(view);
            } else if (message.field == 4 && message.wire == 2) {
                if (!message.bytes(view)) return false;
                Value value;
                if (!decodeValue(view, value)) return false;
                layer.values.push_back(value);
            } else if (message.field == 5 && message.wire == 0) {
                if (!message.varint(raw)) return false;
                layer.extent = raw > 0 ? static_cast<F64>(raw) : 4096.0;
            } else if (!message.skip()) {
                return false;
            }
        }
        if (!message.good()) return false;

        const auto& name = layer.name;
        if (name != "water" && name != "landuse" && name != "waterway" &&
            name != "transportation" && name != "boundary" && name != "place" &&
            name != "aeroway" && name != "aerodrome_label") {
            return true;
        }

        const auto classKey = layer.key("class");
        const auto subclassKey = layer.key("subclass");
        const auto brunnelKey = layer.key("brunnel");
        const auto adminKey = layer.key("admin_level");
        const auto maritimeKey = layer.key("maritime");
        const auto latinKey = layer.key("name:latin");
        const auto nameKey = layer.key("name");
        const auto iataKey = layer.key("iata");
        const auto icaoKey = layer.key("icao");

        for (const auto& featureData : layer.features) {
            Feature feature;
            if (!decodeFeature(featureData, feature)) return false;
            auto attribute = [&](const std::optional<U32>& key) -> const Value* {
                if (!key) return nullptr;
                for (U64 i = 0; i + 1 < feature.tags.size(); i += 2) {
                    if (feature.tags[i] == *key && feature.tags[i + 1] < layer.values.size()) {
                        return &layer.values[feature.tags[i + 1]];
                    }
                }
                return nullptr;
            };
            auto text = [&](const std::optional<U32>& key) -> std::string_view {
                const auto* value = attribute(key);
                return value && value->isText ? value->text : std::string_view{};
            };
            auto number = [&](const std::optional<U32>& key) -> F64 {
                const auto* value = attribute(key);
                return value && !value->isText ? value->number : 0.0;
            };

            const auto kind = text(classKey);
            const bool tunnel = text(brunnelKey) == "tunnel";

            if (name == "water") {
                if (feature.type != 3 || tunnel) continue;
                if (!decodeGeometry(feature.geometry, feature.type)) return false;
                if (!addPolygons(Fill::Water, layer.extent,
                                 kind == "ocean" ? &oceanEdges : &waterEdges)) return false;
            } else if (name == "landuse") {
                if (feature.type != 3) continue;
                if (kind != "residential" && kind != "commercial" &&
                    kind != "industrial" && kind != "retail") continue;
                if (!decodeGeometry(feature.geometry, feature.type)) return false;
                if (!addPolygons(Fill::Urban, layer.extent, nullptr)) return false;
            } else if (name == "waterway") {
                if (feature.type != 2 || tunnel) continue;
                std::optional<Line> line;
                if (kind == "river") line = Line::River;
                else if (kind == "canal") line = Line::Canal;
                else if (kind == "stream") line = Line::Stream;
                if (!line) continue;
                if (!decodeGeometry(feature.geometry, feature.type)) return false;
                addLines(*line, layer.extent);
            } else if (name == "transportation") {
                if (feature.type != 2 || tunnel) continue;
                std::optional<Line> line;
                if (kind == "motorway" || kind == "trunk") line = Line::Motorway;
                else if (kind == "primary") line = Line::PrimaryRoad;
                else if (kind == "secondary" || kind == "tertiary") line = Line::SecondaryRoad;
                else if (kind == "minor") line = Line::MinorRoad;
                else if (kind == "rail" && text(subclassKey) != "subway") line = Line::Rail;
                if (!line) continue;
                if (!decodeGeometry(feature.geometry, feature.type)) return false;
                addLines(*line, layer.extent);
            } else if (name == "boundary") {
                if (feature.type != 2 || number(maritimeKey) != 0.0) continue;
                const F64 level = number(adminKey);
                std::optional<Line> line;
                if (level == 2.0) line = Line::CountryBorder;
                else if (level == 3.0 || level == 4.0) line = Line::StateBorder;
                if (!line) continue;
                if (!decodeGeometry(feature.geometry, feature.type)) return false;
                addLines(*line, layer.extent);
            } else if (name == "aeroway") {
                if (feature.type == 3) {
                    if (kind != "aerodrome" && kind != "apron" && kind != "runway" &&
                        kind != "taxiway") continue;
                    if (!decodeGeometry(feature.geometry, feature.type)) return false;
                    if (!addPolygons(Fill::Airport, layer.extent, nullptr)) return false;
                    continue;
                }
                if (feature.type != 2) continue;
                std::optional<Line> line;
                if (kind == "runway") line = Line::Runway;
                else if (kind == "taxiway") line = Line::Taxiway;
                if (!line) continue;
                if (!decodeGeometry(feature.geometry, feature.type)) return false;
                addLines(*line, layer.extent);
            } else if (name == "place" || name == "aerodrome_label") {
                if (feature.type != 1) continue;
                std::optional<Place> place;
                std::string_view label;
                if (name == "aerodrome_label") {
                    label = text(iataKey);
                    if (!RenderableName(label)) label = text(icaoKey);
                    place = kind == "international" ? Place::Airport : Place::Airfield;
                } else {
                    if (kind == "city") place = Place::City;
                    else if (kind == "town") place = Place::Town;
                    else if (kind == "village") place = Place::Village;
                    else if (kind == "suburb") place = Place::Suburb;
                    else if (kind == "hamlet" || kind == "quarter" || kind == "neighbourhood") place = Place::Hamlet;
                    label = text(latinKey);
                    if (!RenderableName(label)) label = text(nameKey);
                }
                if (!place || !RenderableName(label)) continue;
                if (!decodeGeometry(feature.geometry, feature.type)) return false;
                if (parts.empty()) continue;
                const auto& point = parts.front().front();
                if (point[0] < 0 || point[1] < 0 || point[0] >= layer.extent || point[1] >= layer.extent) {
                    continue;
                }
                tile->labels.push_back({
                    .lon = static_cast<F32>(point[0] / layer.extent),
                    .lat = static_cast<F32>(point[1] / layer.extent),
                    .kind = *place,
                    .name = std::string(label),
                });
            }
        }
        if (name == "water") addCoastline(layer.extent);
        return true;
    }
};

Platform::Http::Headers TileHeaders() {
#if defined(JST_OS_BROWSER)
    return {};
#else
    return {{"User-Agent", std::string("CyberEther/") + JETSTREAM_VERSION_STR}, {"Accept-Encoding", "gzip"}};
#endif
}

}  // namespace

U64 GeoMapTiles::Key::id() const {
    return (static_cast<U64>(z) << 58) | (static_cast<U64>(x) << 29) | static_cast<U64>(y);
}

//
// OpenFreeMap
//

struct GeoMapTiles::OpenFreeMap::Impl {
    static constexpr const char* Host = "https://tiles.openfreemap.org";

    std::mutex mutex;
    std::string host;
    std::string pathTemplate;
    Clock::time_point retryAt{};

    bool resolve(const std::function<bool()>& cancelled) {
        std::lock_guard lock(mutex);
        if (!pathTemplate.empty()) return true;
        if (Clock::now() < retryAt) return false;

        Platform::Http::Client::Request request;
        request.host = Host;
        request.path = "/planet";
        request.headers = TileHeaders();
        request.timeoutSeconds = 15;
        request.cancelled = cancelled;
        std::string body;
        const auto response = Download(request, body);
        if (!response.error && response.status == 200 && Inflate(body)) {
            const auto tiles = body.find("\"tiles\"");
            const auto open = tiles == std::string::npos ? tiles : body.find('[', tiles);
            const auto begin = open == std::string::npos ? open : body.find('"', open);
            const auto end = begin == std::string::npos ? begin : body.find('"', begin + 1);
            if (end != std::string::npos) {
                std::string url = body.substr(begin + 1, end - begin - 1);
                for (auto at = url.find("\\/"); at != std::string::npos; at = url.find("\\/", at)) {
                    url.erase(at, 1);
                }
                const auto scheme = url.find("://");
                const auto slash = scheme == std::string::npos ? scheme : url.find('/', scheme + 3);
                if (slash != std::string::npos && url.find("{z}") != std::string::npos) {
                    host = url.substr(0, slash);
                    pathTemplate = url.substr(slash);
                    return true;
                }
            }
        }
        JST_WARN("[GEOMAP] Could not resolve the OpenFreeMap tile manifest.");
        retryAt = Clock::now() + std::chrono::seconds(30);
        return false;
    }
};

GeoMapTiles::OpenFreeMap::OpenFreeMap() : pimpl(std::make_unique<Impl>()) {}

GeoMapTiles::OpenFreeMap::~OpenFreeMap() = default;

std::string GeoMapTiles::OpenFreeMap::name() const {
    return "openfreemap";
}

std::string GeoMapTiles::OpenFreeMap::attribution() const {
    return "OpenFreeMap (c) OpenMapTiles Data from OpenStreetMap";
}

U8 GeoMapTiles::OpenFreeMap::maxZoom() const {
    return 14;
}

GeoMapTiles::Source::Status GeoMapTiles::OpenFreeMap::fetch(const Key& key,
                                                            std::string& data,
                                                            const std::function<bool()>& cancelled) {
    if (!pimpl->resolve(cancelled)) return Status::Failed;

    std::string path = pimpl->pathTemplate;
    auto replace = [&](std::string_view token, U32 value) {
        const auto at = path.find(token);
        if (at != std::string::npos) path.replace(at, token.size(), std::to_string(value));
    };
    replace("{z}", key.z);
    replace("{x}", key.x);
    replace("{y}", key.y);

    Platform::Http::Client::Request request;
    request.host = pimpl->host;
    request.path = path;
    request.headers = TileHeaders();
    request.timeoutSeconds = 20;
    request.cancelled = cancelled;
    std::string body;
    const auto response = Download(request, body);
    if (response.error) return Status::Failed;
    if (response.status == 204 || response.status == 404) return Status::Missing;
    if (response.status != 200) return Status::Failed;
    data = std::move(body);
    return Status::Found;
}

//
// GeoMapTiles
//

struct GeoMapTiles::Impl {
    struct Entry {
        enum class State { Loading, Ready, Failed };

        State state = State::Loading;
        std::shared_ptr<const TileData> tile;
        Clock::time_point retryAt{};
        U32 failures = 0;
        U64 lastUsed = 0;
    };

    struct Piece {
        Key target;
        Key source;
        std::shared_ptr<const TileData> tile;
        Geometry geometry;
        F32 born = 0.0f;
    };

    struct Fading {
        Piece piece;
        F32 died = 0.0f;
    };

    std::unique_ptr<Source> source;
    Config config;
    std::filesystem::path cacheDirectory;
    bool diskCache = false;
    std::atomic<U64> diskWritten{0};
    std::atomic<bool> pruning{false};

    std::mutex mutex;
    std::condition_variable wake;
    std::atomic<bool> stopping{false};
    std::atomic<bool> warned{false};
    std::atomic<U64> completed{0};
    std::unordered_map<U64, Entry> entries;
    std::vector<Key> queue;
    std::vector<std::unique_ptr<Platform::WorkerThread>> workers;
    std::unique_ptr<Platform::WorkerThread> assembler;
    std::condition_variable assembleWake;
    bool assembleRequested = false;
    std::vector<Key> assembleTargets;
    std::shared_ptr<const Geometry> published = std::make_shared<Geometry>();
    F32 publishedSettled = 0.0f;
    const Clock::time_point epoch = Clock::now();

    std::optional<MapContext::Uniforms> lastView;
    std::optional<U8> level;
    F32 fade = 0.0f;
    F32 zoom = 0.0f;
    std::vector<Key> wanted;
    bool wantedChanged = false;
    U64 requestedCompletions = 0;
    Clock::time_point lastRequest{};
    std::shared_ptr<const Geometry> current = published;
    F32 currentSettled = 0.0f;

    U64 tick = 0;
    U64 revision = 0;
    std::map<std::pair<U64, U64>, Piece> pieces;
    std::vector<Fading> fading;
    F32 settled = 0.0f;

    F32 seconds() const {
        return std::chrono::duration<F32>(Clock::now() - epoch).count();
    }

    std::filesystem::path tilePath(const Key& key) const {
        return cacheDirectory / std::to_string(key.z) / std::to_string(key.x) /
               (std::to_string(key.y) + ".mvt");
    }

    bool readDisk(const Key& key, std::string& data, bool& stale) const {
        if (!diskCache) return false;
        std::error_code ec;
        const auto path = tilePath(key);
        const auto size = std::filesystem::file_size(path, ec);
        if (ec || size == 0 || size > MaxInflatedBytes) return false;
        const auto modified = std::filesystem::last_write_time(path, ec);
        if (ec) return false;
        stale = std::filesystem::file_time_type::clock::now() - modified > DiskMaxAge;
        std::ifstream file(path, std::ios::binary);
        if (!file) return false;
        data.resize(size);
        file.read(data.data(), static_cast<std::streamsize>(size));
        return static_cast<U64>(file.gcount()) == size;
    }

    void writeDisk(const Key& key, const std::string& data) {
        if (!diskCache) return;
        std::error_code ec;
        const auto path = tilePath(key);
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) return;
        auto temporary = path;
        temporary += ".tmp";
        {
            std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
            if (!file) return;
            file.write(data.data(), static_cast<std::streamsize>(data.size()));
            if (!file) {
                file.close();
                std::filesystem::remove(temporary, ec);
                return;
            }
        }
        std::filesystem::rename(temporary, path, ec);
        if (ec) {
            std::filesystem::remove(temporary, ec);
            return;
        }
        diskWritten += data.size();
    }

    void prune() {
        if (diskWritten < config.diskBudgetBytes / 8 || pruning.exchange(true)) return;
        diskWritten = 0;
        pruneDisk();
        pruning = false;
    }

    void pruneDisk() const {
        if (!diskCache) return;
        struct File {
            std::filesystem::path path;
            U64 size;
            std::filesystem::file_time_type modified;
        };
        std::vector<File> files;
        U64 total = 0;
        std::error_code ec;
        for (auto it = std::filesystem::recursive_directory_iterator(cacheDirectory, ec);
             !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
            if (stopping) return;
            if (!it->is_regular_file(ec)) continue;
            const U64 size = it->file_size(ec);
            const auto modified = it->last_write_time(ec);
            if (ec) {
                ec.clear();
                continue;
            }
            files.push_back({it->path(), size, modified});
            total += size;
        }
        if (total <= config.diskBudgetBytes) return;
        std::sort(files.begin(), files.end(), [](const File& a, const File& b) {
            return a.modified < b.modified;
        });
        const U64 target = config.diskBudgetBytes / 4 * 3;
        for (const auto& file : files) {
            if (total <= target) break;
            if (std::filesystem::remove(file.path, ec)) total -= file.size;
        }
    }

    std::shared_ptr<const TileData> decode(std::string& data) const {
        if (!Inflate(data)) return nullptr;
        Decoder decoder;
        return decoder.run(data);
    }

    std::optional<std::shared_ptr<const TileData>> load(const Key& key) {
        std::string cached;
        bool stale = false;
        const bool hit = readDisk(key, cached, stale);
        if (hit && !stale) {
            std::string copy = cached;
            if (auto tile = decode(copy)) return tile;
            std::error_code ec;
            std::filesystem::remove(tilePath(key), ec);
            cached.clear();
        }

        std::string body;
        const auto status = source->fetch(key, body, [this] { return stopping.load(); });
        if (status == Source::Status::Missing) {
            if (!cached.empty()) {
                if (auto tile = decode(cached)) return tile;
            }
            auto tile = std::make_shared<TileData>();
            tile->missing = true;
            return tile;
        }
        if (status == Source::Status::Found) {
            std::string copy = body;
            if (auto tile = decode(copy)) {
                writeDisk(key, body);
                return tile;
            }
            JST_DEBUG("[GEOMAP] Tile {}/{}/{} could not be decoded.", key.z, key.x, key.y);
        }
        if (!cached.empty()) {
            if (auto tile = decode(cached)) return tile;
        }
        if (!stopping && !warned.exchange(true)) {
            JST_WARN("[GEOMAP] Detail tiles are unavailable, showing Natural Earth only.");
        }
        return std::nullopt;
    }

    bool pick(Key& key) {
        const auto now = Clock::now();
        for (const auto& candidate : queue) {
            auto [it, inserted] = entries.try_emplace(candidate.id());
            if (inserted) {
                key = candidate;
                return true;
            }
            auto& entry = it->second;
            if (entry.state == Entry::State::Failed && now >= entry.retryAt) {
                entry.state = Entry::State::Loading;
                key = candidate;
                return true;
            }
        }
        return false;
    }

    void work() {
        while (true) {
            prune();
            Key key;
            {
                std::unique_lock lock(mutex);
                while (!stopping && !pick(key)) {
                    wake.wait_for(lock, std::chrono::seconds(1));
                }
                if (stopping) return;
            }
            auto tile = load(key);
            {
                std::lock_guard lock(mutex);
                auto& entry = entries[key.id()];
                if (tile) {
                    entry.state = Entry::State::Ready;
                    entry.tile = std::move(*tile);
                    entry.failures = 0;
                } else {
                    entry.state = Entry::State::Failed;
                    entry.failures = std::min<U32>(entry.failures + 1, 4);
                    entry.retryAt = Clock::now() + std::chrono::seconds(15u << (entry.failures - 1));
                }
            }
            ++completed;
        }
    }

    void select(const MapContext& context) {
        zoom = DetailZoom(context, config.tilePixels);
        fade = glm::smoothstep(config.fadeStart, std::max(config.fadeEnd, config.fadeStart + 0.01f), zoom);

        std::vector<Key> next;
        std::vector<Key> previews;
        if (zoom >= config.fadeStart - 0.5f) {
            const F32 maxZoom = static_cast<F32>(source->maxZoom());
            if (!level || zoom > *level + 0.65f || zoom < *level - 0.65f) {
                level = static_cast<U8>(std::clamp(std::round(zoom), 0.0f, maxZoom));
            }
            const auto& view = context.view;
            const glm::dvec3 normal(MapContext::LonLatToSphere(view.centerLon, view.centerLat));
            const glm::dvec3 up(MapContext::CameraUp(MapContext::LonLatToSphere(view.centerLon,
                                                                                   view.centerLat)));
            const glm::dvec3 eye = normal * (1.0 + BaseHeight / std::pow(2.0, static_cast<F64>(view.zoom)));
            const glm::dvec3 forward = -normal;
            const glm::dvec3 right = glm::normalize(glm::cross(forward, up));
            const glm::dvec3 upward = glm::cross(right, forward);
            const F64 aspect = view.aspectRatio;

            auto tileOf = [&](U8 z, F64 lon, F64 lat) {
                const U32 n = 1u << z;
                const F64 x = (lon + 180.0) / 360.0 * n;
                const F64 y = LatitudeToY(lat) * n;
                return Key{z, std::min(static_cast<U32>(std::max(x, 0.0)), n - 1),
                           std::min(static_cast<U32>(std::max(y, 0.0)), n - 1)};
            };

            auto sample = [&](U8 z) {
                std::unordered_set<U64> seen;
                std::vector<Key> hits;
                for (U32 i = 0; i < SelectionSamples; ++i) {
                    for (U32 j = 0; j < SelectionSamples; ++j) {
                        const F64 nx = -1.0 + 2.0 * i / (SelectionSamples - 1);
                        const F64 ny = -1.0 + 2.0 * j / (SelectionSamples - 1);
                        const glm::dvec3 direction = glm::normalize(
                            forward + right * (nx * HalfFovTan * aspect) + upward * (ny * HalfFovTan));
                        const F64 b = glm::dot(eye, direction);
                        const F64 c = glm::dot(eye, eye) - 1.0;
                        const F64 discriminant = b * b - c;
                        if (discriminant < 0.0) continue;
                        const F64 t = -b - std::sqrt(discriminant);
                        if (t < 0.0) continue;
                        const glm::dvec3 p = eye + direction * t;
                        const F64 lat = std::asin(std::clamp(p.y, -1.0, 1.0)) * 180.0 / Pi;
                        const F64 lon = std::atan2(p.x, p.z) * 180.0 / Pi;
                        if (std::abs(lat) > MaxLatitude) continue;
                        const Key key = tileOf(z, lon, lat);
                        if (seen.insert(key.id()).second) hits.push_back(key);
                    }
                }
                return hits;
            };

            auto cover = [&](U8 z, std::vector<Key>& ring) {
                std::vector<Key> keys;
                ring.clear();
                const auto hits = sample(z);
                if (hits.empty()) return keys;
                const I64 n = static_cast<I64>(1) << z;
                const Key center = tileOf(z, view.centerLon, view.centerLat);
                auto offset = [&](const Key& key) {
                    const I64 d = static_cast<I64>(key.x) - static_cast<I64>(center.x);
                    return ((d % n) + n + n / 2) % n - n / 2;
                };
                I64 x0 = offset(hits.front());
                I64 x1 = x0;
                I64 y0 = hits.front().y;
                I64 y1 = y0;
                for (const auto& hit : hits) {
                    x0 = std::min(x0, offset(hit));
                    x1 = std::max(x1, offset(hit));
                    y0 = std::min<I64>(y0, hit.y);
                    y1 = std::max<I64>(y1, hit.y);
                }
                x1 = std::min(x1, x0 + n - 1);
                std::unordered_set<U64> seen;
                auto add = [&](std::vector<Key>& out, I64 dx, I64 y) {
                    if (y < 0 || y >= n) return;
                    const I64 x = ((static_cast<I64>(center.x) + dx) % n + n) % n;
                    const Key key{z, static_cast<U32>(x), static_cast<U32>(y)};
                    if (seen.insert(key.id()).second) out.push_back(key);
                };
                for (I64 y = y0; y <= y1; ++y) {
                    for (I64 dx = x0; dx <= x1; ++dx) add(keys, dx, y);
                }
                for (I64 y = y0 - 1; y <= y1 + 1; ++y) {
                    for (I64 dx = x0 - 1; dx <= x1 + 1; ++dx) add(ring, dx, y);
                }
                return keys;
            };

            std::vector<Key> ring;
            auto hits = cover(*level, ring);
            while (hits.size() + ring.size() > config.maxTiles && *level > 0) {
                level = static_cast<U8>(*level - 1);
                hits = cover(*level, ring);
            }
            const U8 z = *level;
            const U32 n = 1u << z;

            const Key center = tileOf(z, view.centerLon, view.centerLat);
            auto distance = [&](const Key& key) {
                const I64 dx = std::abs(static_cast<I64>(key.x) - static_cast<I64>(center.x));
                const I64 dy = static_cast<I64>(key.y) - static_cast<I64>(center.y);
                const I64 wrapped = std::min<I64>(dx, static_cast<I64>(n) - dx);
                return wrapped * wrapped + dy * dy;
            };
            auto closer = [&](const Key& a, const Key& b) {
                return distance(a) < distance(b);
            };
            std::stable_sort(hits.begin(), hits.end(), closer);
            std::stable_sort(ring.begin(), ring.end(), closer);
            if (z >= PreviewLevels) {
                std::unordered_set<U64> added;
                for (const auto& hit : hits) {
                    const Key preview{static_cast<U8>(z - PreviewLevels), hit.x >> PreviewLevels,
                                      hit.y >> PreviewLevels};
                    if (added.insert(preview.id()).second) previews.push_back(preview);
                }
            }
            next = std::move(hits);
            next.insert(next.end(), ring.begin(), ring.end());
        } else {
            level.reset();
        }

        if (next != wanted) {
            wanted = std::move(next);
            wantedChanged = true;
            {
                std::lock_guard lock(mutex);
                queue = previews;
                queue.insert(queue.end(), wanted.begin(), wanted.end());
            }
            wake.notify_all();
        }
    }

    Piece build(const Key& target, const Key& source, std::shared_ptr<const TileData> tile) const {
        Piece piece;
        piece.target = target;
        piece.source = source;
        piece.tile = tile;
        auto& out = piece.geometry;

        const bool clip = !(target == source);
        Rect rect;
        if (clip) {
            const U32 depth = target.z - source.z;
            const F64 span = 1.0 / static_cast<F64>(1u << depth);
            rect.x0 = (target.x - (static_cast<U64>(source.x) << depth)) * span;
            rect.y0 = (target.y - (static_cast<U64>(source.y) << depth)) * span;
            rect.x1 = rect.x0 + span;
            rect.y1 = rect.y0 + span;
        }

        const F64 sourceScale = 1.0 / static_cast<F64>(1u << source.z);
        auto lonOf = [&](F64 u) {
            return static_cast<F32>((source.x + u) * sourceScale * 360.0 - 180.0);
        };
        auto latOf = [&](F64 v) {
            return static_cast<F32>(YToLatitude((source.y + v) * sourceScale));
        };

        for (U64 c = 0; c < LineCount; ++c) {
            const auto& in = tile->lines[c];
            auto& dst = out.lines[c];
            dst.reserve(in.size());
            for (U64 i = 0; i + 3 < in.size(); i += 4) {
                F64 x0 = in[i];
                F64 y0 = in[i + 1];
                F64 x1 = in[i + 2];
                F64 y1 = in[i + 3];
                if (clip && !ClipSegment(x0, y0, x1, y1, rect)) continue;
                dst.insert(dst.end(), {lonOf(x0), latOf(y0), lonOf(x1), latOf(y1)});
            }
        }

        for (U64 f = 0; f < FillCount; ++f) {
            const auto& in = tile->fills[f];
            auto& dst = out.fills[f];
            if (in.indices.empty()) continue;
            auto emit = [&](F64 u, F64 v) {
                dst.positions.push_back(lonOf(u));
                dst.positions.push_back(latOf(v));
                return static_cast<U32>(dst.positions.size() / 2 - 1);
            };
            if (clip) {
                ClipTriangles(in.positions, in.indices, rect, emit, dst.indices);
            } else {
                dst.positions.reserve(in.positions.size());
                for (U64 i = 0; i + 1 < in.positions.size(); i += 2) {
                    emit(in.positions[i], in.positions[i + 1]);
                }
                dst.indices = in.indices;
            }
        }

        {
            auto& land = out.fills[static_cast<U64>(Fill::Land)];
            const F64 targetScale = 1.0 / static_cast<F64>(1u << target.z);
            for (U32 row = 0; row <= FootprintCells; ++row) {
                const F64 v = (target.y + static_cast<F64>(row) / FootprintCells) * targetScale;
                for (U32 col = 0; col <= FootprintCells; ++col) {
                    const F64 u = (target.x + static_cast<F64>(col) / FootprintCells) * targetScale;
                    land.positions.push_back(static_cast<F32>(u * 360.0 - 180.0));
                    land.positions.push_back(static_cast<F32>(YToLatitude(v)));
                }
            }
            for (U32 row = 0; row < FootprintCells; ++row) {
                for (U32 col = 0; col < FootprintCells; ++col) {
                    const U32 a = row * (FootprintCells + 1) + col;
                    const U32 b = a + 1;
                    const U32 c = a + FootprintCells + 1;
                    const U32 d = c + 1;
                    land.indices.insert(land.indices.end(), {a, c, b, b, c, d});
                }
            }
        }

        for (const auto& label : tile->labels) {
            if (clip && !rect.contains(label.lon, label.lat)) continue;
            out.labels.push_back({lonOf(label.lon), latOf(label.lat), label.kind, label.name});
        }

        return piece;
    }

    void assembleLoop() {
        while (true) {
            std::vector<Key> targets;
            {
                std::unique_lock lock(mutex);
                const auto requested = [&] { return stopping || assembleRequested; };
                std::optional<F32> expiry;
                for (const auto& item : fading) {
                    if (item.died >= 0.0f) expiry = std::min(expiry.value_or(item.died), item.died);
                }
                if (!expiry) {
                    assembleWake.wait(lock, requested);
                } else {
                    const auto deadline = epoch + std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<F32>(*expiry + config.fadeSeconds));
                    assembleWake.wait_until(lock, deadline, requested);
                }
                if (stopping) return;
                targets = assembleTargets;
                assembleRequested = false;
            }
            auto next = std::make_shared<const Geometry>(assemble(targets));
            std::lock_guard lock(mutex);
            published = std::move(next);
            publishedSettled = settled;
        }
    }

    Geometry assemble(const std::vector<Key>& targets) {
        ++tick;
        struct Plan {
            Key target;
            Key source;
            std::shared_ptr<const TileData> tile;
        };
        std::vector<Plan> plan;
        {
            std::lock_guard lock(mutex);
            auto ready = [&](const Key& key) -> std::shared_ptr<const TileData> {
                const auto it = entries.find(key.id());
                if (it == entries.end()) return nullptr;
                it->second.lastUsed = tick;
                const auto& entry = it->second;
                if (entry.state != Entry::State::Ready || !entry.tile || entry.tile->missing) return nullptr;
                return entry.tile;
            };
            for (const auto& target : targets) {
                if (auto tile = ready(target)) {
                    plan.push_back({target, target, std::move(tile)});
                    continue;
                }
                bool found = false;
                for (U8 up = 1; up <= MaxFallbackLevels && up <= target.z && !found; ++up) {
                    const Key ancestor{static_cast<U8>(target.z - up), target.x >> up, target.y >> up};
                    if (auto tile = ready(ancestor)) {
                        plan.push_back({target, ancestor, std::move(tile)});
                        found = true;
                    }
                }
                if (found || target.z >= source->maxZoom()) continue;
                std::array<Plan, 4> children;
                bool complete = true;
                for (U32 i = 0; i < 4 && complete; ++i) {
                    const Key child{static_cast<U8>(target.z + 1), target.x * 2 + (i & 1), target.y * 2 + (i >> 1)};
                    auto tile = ready(child);
                    complete = tile != nullptr;
                    children[i] = {child, child, std::move(tile)};
                }
                if (complete) plan.insert(plan.end(), children.begin(), children.end());
            }
        }

        const F32 now = seconds();
        auto overlaps = [](const Key& a, const Key& b) {
            const U8 z = std::min(a.z, b.z);
            return (a.x >> (a.z - z)) == (b.x >> (b.z - z)) &&
                   (a.y >> (a.z - z)) == (b.y >> (b.z - z));
        };
        auto shown = [&](const std::map<std::pair<U64, U64>, Piece>& set, const Piece& piece) -> const Piece* {
            for (const auto& [id, other] : set) {
                if (other.tile && other.tile == piece.tile && overlaps(other.target, piece.target)) {
                    return &other;
                }
            }
            return nullptr;
        };

        std::map<std::pair<U64, U64>, Piece> next;
        for (auto& item : plan) {
            const auto id = std::make_pair(item.target.id(), item.source.id());
            if (next.contains(id)) continue;
            auto previous = pieces.find(id);
            if (previous != pieces.end() && previous->second.tile == item.tile) {
                next.emplace(id, std::move(previous->second));
                pieces.erase(previous);
                continue;
            }
            Piece piece = build(item.target, item.source, item.tile);
            const Piece* earlier = shown(pieces, piece);
            if (!earlier) earlier = shown(next, piece);
            piece.born = earlier ? earlier->born : now;
            next.emplace(id, std::move(piece));
        }

        auto fillSize = [](const Piece& piece) {
            U64 vertices = 0;
            U64 indices = 0;
            for (const auto& fill : piece.geometry.fills) {
                vertices += fill.positions.size() / 2;
                indices += fill.indices.size();
            }
            return std::make_pair(vertices, indices);
        };
        U64 vertices = 0;
        U64 indices = 0;
        auto fits = [&](const Piece& piece) {
            const auto [pieceVertices, pieceIndices] = fillSize(piece);
            if (vertices + pieceVertices > config.fillVertexCapacity ||
                indices + pieceIndices > config.fillIndexCapacity) {
                return false;
            }
            vertices += pieceVertices;
            indices += pieceIndices;
            return true;
        };

        std::vector<const Piece*> visible;
        std::array<U64, LineCount> reserved{};
        std::unordered_set<U64> placed;
        for (const auto& item : plan) {
            const auto id = std::make_pair(item.target.id(), item.source.id());
            if (!placed.insert(item.target.id()).second) continue;
            const auto& piece = next.at(id);
            if (!fits(piece)) continue;
            visible.push_back(&piece);
            for (U64 c = 0; c < LineCount; ++c) reserved[c] += piece.geometry.lines[c].size();
        }

        std::unordered_map<U64, const TileData*> drawn;
        U8 deepest = 0;
        for (const auto* piece : visible) {
            drawn.emplace(piece->target.id(), piece->tile.get());
            deepest = std::max(deepest, piece->target.z);
        }
        auto covered = [&](const Key& key, const TileData* only) {
            auto check = [&](auto& self, const Key& at) -> bool {
                for (U8 up = 0; up <= at.z; ++up) {
                    const Key ancestor{static_cast<U8>(at.z - up), at.x >> up, at.y >> up};
                    const auto it = drawn.find(ancestor.id());
                    if (it != drawn.end() && (!only || it->second == only)) return true;
                }
                if (at.z >= deepest) return false;
                for (U32 i = 0; i < 4; ++i) {
                    const Key child{static_cast<U8>(at.z + 1), at.x * 2 + (i & 1), at.y * 2 + (i >> 1)};
                    if (!self(self, child)) return false;
                }
                return true;
            };
            return check(check, key);
        };

        for (auto& [id, piece] : pieces) {
            if (covered(piece.target, piece.tile.get())) continue;
            fading.push_back({std::move(piece), -1.0f});
        }
        for (auto& item : fading) {
            if (item.died >= 0.0f) continue;
            const auto& target = item.piece.target;
            const bool inView = std::any_of(targets.begin(), targets.end(), [&](const Key& key) {
                return overlaps(key, target);
            });
            if (!inView || covered(target, nullptr)) item.died = now;
        }
        std::erase_if(fading, [&](const Fading& item) {
            return item.died >= 0.0f && now - item.died >= config.fadeSeconds;
        });

        Geometry result;
        result.revision = ++revision;
        settled = 0.0f;
        auto append = [&](const Piece& piece, F32 died, const std::array<U64, LineCount>& keep) {
            const auto& geometry = piece.geometry;
            for (U64 f = 0; f < FillCount; ++f) {
                auto& dst = result.fills[f];
                const auto& src = geometry.fills[f];
                const U32 base = static_cast<U32>(dst.positions.size() / 2);
                dst.positions.insert(dst.positions.end(), src.positions.begin(), src.positions.end());
                for (const U32 index : src.indices) dst.indices.push_back(base + index);
                for (U64 v = 0; v < src.positions.size() / 2; ++v) {
                    result.fillFades[f].insert(result.fillFades[f].end(), {piece.born, died});
                }
            }
            for (U64 c = 0; c < LineCount; ++c) {
                auto& dst = result.lines[c];
                const auto& src = geometry.lines[c];
                const U64 capacity = config.lineCapacity[c] * 4;
                const U64 room = capacity - std::min<U64>(dst.size() + keep[c], capacity);
                const U64 count = std::min<U64>(src.size(), room) / 4 * 4;
                dst.insert(dst.end(), src.begin(), src.begin() + count);
                for (U64 s = 0; s < count / 4; ++s) {
                    result.lineFades[c].insert(result.lineFades[c].end(), {piece.born, died});
                }
            }
            settled = std::max(settled, std::max(piece.born, died) + config.fadeSeconds);
        };

        for (const auto& item : fading) {
            if (!fits(item.piece)) continue;
            append(item.piece, item.died, reserved);
            if (item.died < 0.0f) {
                result.labels.insert(result.labels.end(), item.piece.geometry.labels.begin(),
                                     item.piece.geometry.labels.end());
            }
        }
        for (U64 f = 0; f < FillCount; ++f) result.heldFillIndices[f] = result.fills[f].indices.size();
        const std::array<U64, LineCount> none{};
        for (const auto* piece : visible) {
            append(*piece, -1.0f, none);
            result.labels.insert(result.labels.end(), piece->geometry.labels.begin(),
                                 piece->geometry.labels.end());
        }

        pieces = std::move(next);
        evict();
        return result;
    }

    void evict() {
        std::lock_guard lock(mutex);
        U64 total = 0;
        std::vector<std::pair<U64, U64>> candidates;
        std::vector<U64> failed;
        for (const auto& [id, entry] : entries) {
            if (entry.state == Entry::State::Ready) {
                total += entry.tile ? entry.tile->bytes() : 0;
                if (entry.lastUsed < tick) candidates.emplace_back(entry.lastUsed, id);
            } else if (entry.state == Entry::State::Failed && entry.lastUsed + 64 < tick) {
                failed.push_back(id);
            }
        }
        for (const U64 id : failed) entries.erase(id);
        if (total <= config.memoryBudgetBytes) return;
        std::sort(candidates.begin(), candidates.end());
        for (const auto& [lastUsed, id] : candidates) {
            if (total <= config.memoryBudgetBytes) break;
            const auto it = entries.find(id);
            total -= it->second.tile ? it->second.tile->bytes() : 0;
            entries.erase(it);
        }
    }
};

GeoMapTiles::GeoMapTiles(std::unique_ptr<Source> source) : pimpl(std::make_unique<Impl>()) {
    pimpl->source = std::move(source);
}

GeoMapTiles::~GeoMapTiles() {
    stop();
}

void GeoMapTiles::start(const Config& config) {
    if (!pimpl->workers.empty() || !pimpl->source) return;
    pimpl->config = config;
    pimpl->stopping = false;
    pimpl->lastView.reset();
    // stop() clears published geometry but keeps cached tiles and the selection.
    // Rebuild even when the first view and completion count are unchanged.
    pimpl->wantedChanged = true;

#if !defined(JST_OS_BROWSER)
    std::string cachePath;
    if (Platform::CachePath(cachePath) == Result::SUCCESS) {
        pimpl->cacheDirectory = Platform::PathFromUtf8(cachePath) / "geomap" / pimpl->source->name();
        pimpl->diskCache = true;
        pimpl->diskWritten = config.diskBudgetBytes;
    }
#endif

    for (U64 i = 0; i < std::max<U64>(config.workers, 1); ++i) {
        auto worker = std::make_unique<Platform::WorkerThread>();
        worker->start([impl = pimpl.get()] { impl->work(); });
        pimpl->workers.push_back(std::move(worker));
    }
    pimpl->assembler = std::make_unique<Platform::WorkerThread>();
    pimpl->assembler->start([impl = pimpl.get()] { impl->assembleLoop(); });
}

void GeoMapTiles::stop() {
    if (pimpl->workers.empty()) return;
    {
        std::lock_guard lock(pimpl->mutex);
        pimpl->stopping = true;
    }
    pimpl->wake.notify_all();
    pimpl->assembleWake.notify_all();
    for (auto& worker : pimpl->workers) {
        if (worker->joinable()) worker->join();
    }
    pimpl->workers.clear();
    if (pimpl->assembler && pimpl->assembler->joinable()) {
        pimpl->assembler->join();
    }
    pimpl->assembler.reset();

    auto empty = std::make_shared<Geometry>();
    empty->revision = ++pimpl->revision;
    pimpl->published = empty;
    pimpl->current = std::move(empty);
    pimpl->pieces.clear();
    pimpl->fading.clear();
    pimpl->publishedSettled = 0.0f;
    pimpl->currentSettled = 0.0f;
}

bool GeoMapTiles::update(const MapContext& context) {
    if (pimpl->workers.empty()) return false;
    if (!pimpl->lastView || *pimpl->lastView != context.view) {
        pimpl->lastView = context.view;
        pimpl->select(context);
    }

    const auto now = Clock::now();
    const U64 completed = pimpl->completed.load();
    const bool arrived = completed != pimpl->requestedCompletions &&
                         now - pimpl->lastRequest >= AssemblyInterval;

    std::lock_guard lock(pimpl->mutex);
    if (pimpl->wantedChanged || arrived) {
        pimpl->wantedChanged = false;
        pimpl->requestedCompletions = completed;
        pimpl->lastRequest = now;
        pimpl->assembleTargets = pimpl->wanted;
        pimpl->assembleRequested = true;
        pimpl->assembleWake.notify_one();
    }
    if (pimpl->published == pimpl->current) {
        return pimpl->seconds() < pimpl->currentSettled + 0.1f;
    }
    pimpl->current = pimpl->published;
    pimpl->currentSettled = pimpl->publishedSettled;
    return true;
}

F32 GeoMapTiles::time() const {
    return pimpl->seconds();
}

F32 GeoMapTiles::opacity() const {
    return pimpl->workers.empty() ? 0.0f : pimpl->fade;
}

F32 GeoMapTiles::zoom() const {
    return pimpl->zoom;
}

const GeoMapTiles::Geometry& GeoMapTiles::geometry() const {
    return *pimpl->current;
}

std::string GeoMapTiles::attribution() const {
    return pimpl->source ? pimpl->source->attribution() : std::string{};
}

F32 GeoMapTiles::DetailZoom(const MapContext& context, F32 tilePixels) {
    const F32 lat = std::clamp(context.view.centerLat, -84.0f, 84.0f);
    const F32 lon = context.view.centerLon;
    constexpr F32 Delta = 0.01f;
    F32 ax = 0.0f;
    F32 ay = 0.0f;
    F32 bx = 0.0f;
    F32 by = 0.0f;
    if (!context.projectLonLat(lon, lat - Delta, ax, ay) ||
        !context.projectLonLat(lon, lat + Delta, bx, by)) {
        return 0.0f;
    }
    const auto pixel = context.pixelSize();
    const F64 dx = (bx - ax) / pixel.x;
    const F64 dy = (by - ay) / pixel.y;
    const F64 pixelsPerDegree = std::sqrt(dx * dx + dy * dy) / (2.0 * Delta);
    if (!(pixelsPerDegree > 0.0) || tilePixels <= 0.0f) return 0.0f;
    return static_cast<F32>(std::log2(360.0 * std::cos(lat * Pi / 180.0) * pixelsPerDegree / tilePixels));
}

namespace {

std::atomic<bool> DetailTilesEnabled{true};

Result CacheRoot(std::filesystem::path& root) {
#if defined(JST_OS_BROWSER)
    (void)root;
    return Result::ERROR;
#else
    std::string cachePath;
    JST_CHECK(Platform::CachePath(cachePath));
    root = Platform::PathFromUtf8(cachePath) / "geomap";
    return Result::SUCCESS;
#endif
}

}  // namespace

void GeoMapTiles::SetEnabled(bool enabled) {
    DetailTilesEnabled = enabled;
}

bool GeoMapTiles::Enabled() {
    return DetailTilesEnabled;
}

Result GeoMapTiles::CacheSize(U64& bytes) {
    bytes = 0;
    std::filesystem::path root;
    JST_CHECK(CacheRoot(root));
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) return Result::SUCCESS;
    for (auto it = std::filesystem::recursive_directory_iterator(root, ec);
         !ec && it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        std::error_code entry;
        if (!it->is_regular_file(entry)) continue;
        const U64 size = it->file_size(entry);
        if (!entry) bytes += size;
    }
    return ec ? Result::ERROR : Result::SUCCESS;
}

Result GeoMapTiles::ClearCache() {
    std::filesystem::path root;
    JST_CHECK(CacheRoot(root));
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    return ec ? Result::ERROR : Result::SUCCESS;
}

}  // namespace Jetstream::Render::Components
