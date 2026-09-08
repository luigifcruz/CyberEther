#include "map_layers.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "jetstream/render/base.hh"
#include "jetstream/render/components/text.hh"
#include "resources/shaders/adsb_shaders.hh"

namespace Jetstream::Modules {

namespace {
using Render::Components::MapContext;
using Render::Components::MapLayer;
using Render::Components::Text;

constexpr ColorRGBA<F32> TargetColor{0.35f, 1.0f, 0.55f, 1.0f};
constexpr ColorRGBA<F32> StaleColor{1.0f, 0.68f, 0.30f, 1.0f};
constexpr F32 LabelScale = 0.60f * AdsbTrackingScale;

std::string LeftAlignedBlock(const std::string& fill) {
    // Text centers individual lines within a multiline block. Right-pad this
    // monospace overlay instead of changing Text (and the map's own labels).
    U64 width = 0;
    for (U64 start = 0; start < fill.size();) {
        const auto end = fill.find('\n', start);
        const U64 length = (end == std::string::npos ? fill.size() : end) - start;
        width = std::max(width, length);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    std::string result;
    for (U64 start = 0; start < fill.size();) {
        const auto end = fill.find('\n', start);
        const U64 length = (end == std::string::npos ? fill.size() : end) - start;
        result.append(fill, start, length);
        result.append(width - length, ' ');
        if (end == std::string::npos) break;
        result += '\n';
        start = end + 1;
    }
    return result;
}

// These are ADS-B presentation resources, not map features. Both batches are
// projected through the shared map context on the CPU, including horizon culling.
struct Batch {
    U64 count = 0;
    std::vector<F32> data;
    std::shared_ptr<Render::Buffer> instances;
    std::shared_ptr<Render::Vertex> vertex;
    std::shared_ptr<Render::Draw> draw;
    std::shared_ptr<Render::Program> program;

    Result create(Render::Window* window, const char* shader, U32 stride, U64 capacity,
                   const std::shared_ptr<Render::Buffer>& quad,
                   const std::shared_ptr<Render::Buffer>& uniforms) {
        data.resize(capacity * stride);
        {
            Render::Buffer::Config cfg;
            cfg.buffer = data.data();
            cfg.size = data.size();
            cfg.elementByteSize = sizeof(F32);
            cfg.target = Render::Buffer::Target::VERTEX;
            cfg.enableZeroCopy = false;
            JST_CHECK(window->build(instances, cfg));
        }
        {
            Render::Vertex::Config cfg;
            cfg.vertices = {{quad, 2}};
            cfg.instances = {{instances, stride}};
            JST_CHECK(window->build(vertex, cfg));
        }
        {
            Render::Draw::Config cfg;
            cfg.buffer = vertex;
            cfg.mode = Render::Draw::Mode::TRIANGLES;
            cfg.numberOfInstances = 0;
            JST_CHECK(window->build(draw, cfg));
        }
        {
            Render::Program::Config cfg;
            cfg.shaders = ShadersPackage[shader];
            cfg.draws = {draw};
            cfg.buffers = {{uniforms, Render::Program::Target::VERTEX}};
            cfg.enableAlphaBlending = true;
            JST_CHECK(window->build(program, cfg));
        }
        return Result::SUCCESS;
    }

    template<size_t N>
    void append(const std::array<F32, N>& instance) {
        if ((count + 1) * N > data.size()) return;
        std::copy(instance.begin(), instance.end(), data.begin() + count++ * N);
    }

    Result present() {
        JST_CHECK(instances->update());
        return draw->updateInstanceCount(count);
    }
};

class RadarLayer final : public MapLayer {
 public:
    explicit RadarLayer(std::shared_ptr<const AdsbMapState> state) : state(std::move(state)) {}

    Result create(Render::Window* window, const MapContext&) override {
        static F32 quad[] = {-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1};
        {
            Render::Buffer::Config cfg;
            cfg.buffer = quad;
            cfg.size = 12;
            cfg.elementByteSize = sizeof(F32);
            cfg.target = Render::Buffer::Target::VERTEX;
            JST_CHECK(window->build(quadBuffer, cfg));
        }
        {
            Render::Buffer::Config cfg;
            cfg.buffer = uniforms.data();
            cfg.size = 1;
            cfg.elementByteSize = sizeof(uniforms);
            cfg.target = Render::Buffer::Target::UNIFORM;
            JST_CHECK(window->build(uniformBuffer, cfg));
        }
        JST_CHECK(targets.create(window, "target", 12, AdsbMapState::MaxAircraft * 7,
                                  quadBuffer, uniformBuffer));
        JST_CHECK(leaders.create(window, "leader", 8, AdsbMapState::MaxAircraft,
                                  quadBuffer, uniformBuffer));
        if (window->hasFont("default_mono")) {
            Text::Config cfg;
            cfg.font = window->font("default_mono");
            cfg.maxCharacters = 64;
            cfg.color = TargetColor;
            ids.resize(AdsbMapState::MaxAircraft);
            for (U64 i = 0; i < ids.size(); ++i) {
                ids[i] = jst::fmt::format("track{:03d}", i);
                cfg.elements[ids[i]] = {.scale = 0, .alignment = {0, 0}};
            }
            JST_CHECK(window->build(text, cfg));
            JST_CHECK(window->bind(text));
            cfg.color = {0.0f, 0.0f, 0.0f, 0.95f};
            JST_CHECK(window->build(shadow, cfg));
            JST_CHECK(window->bind(shadow));
        } else {
            JST_WARN("[MODULE_ADSB] default_mono unavailable; radar data blocks disabled.");
        }
        return Result::SUCCESS;
    }

    Result destroy(Render::Window* window) override {
        if (shadow) JST_CHECK(window->unbind(shadow));
        if (text) JST_CHECK(window->unbind(text));
        shadow.reset();
        text.reset();
        targets = {};
        leaders = {};
        uniformBuffer.reset();
        quadBuffer.reset();
        preferredDirections.clear();
        previousSlots = 0;
        return Result::SUCCESS;
    }

    Result surface(Render::Surface::Config& config) override {
        config.programs.push_back(leaders.program);
        config.programs.push_back(targets.program);
        if (shadow) JST_CHECK(shadow->surface(config));
        if (text) JST_CHECK(text->surface(config));
        return Result::SUCCESS;
    }

    Result present(const MapContext& context) override {
        const auto pixel = context.pixelSize();
        const Extent2D<F32> viewport{2.0f / pixel.x, 2.0f / pixel.y};
        auto toPixels = [&](Extent2D<F32> p) {
            return Extent2D<F32>{(p.x + 1) / pixel.x, (1 - p.y) / pixel.y};
        };
        auto toNdc = [&](Extent2D<F32> p) {
            return Extent2D<F32>{p.x * pixel.x - 1, 1 - p.y * pixel.y};
        };
        std::unordered_set<U32> live;
        const auto visible = AdsbVisibleTargets(state->aircraft, context);
        std::vector<AdsbLabelBox> occupied;
        for (const auto& ac : state->aircraft) {
            if (AdsbTargetVisible(ac)) live.insert(ac.icao);
        }
        for (const auto& target : visible) {
            const auto p = toPixels(target.ndc);
            constexpr F32 radius = 10.0f * AdsbTrackingScale;
            occupied.push_back({p.x - radius, p.y - radius, 2 * radius, 2 * radius});
        }
        std::erase_if(preferredDirections, [&](const auto& item) { return !live.contains(item.first); });
        targets.count = leaders.count = 0;
        U64 slot = 0;
        if (text) {
            JST_CHECK(text->updatePixelSize(pixel));
            JST_CHECK(shadow->updatePixelSize(pixel));
        }
        for (const auto& target : visible) {
            const auto& ac = state->aircraft[target.index];
            const bool stale = !ac.positionAgeSeconds || *ac.positionAgeSeconds >= AdsbStaleSeconds;
            const auto color = stale ? StaleColor : TargetColor;
            auto marker = [&](Extent2D<F32> position, F32 radius, F32 kind, F32 opacity) {
                targets.append<12>({position.x, position.y, radius * AdsbTrackingScale,
                    target.heading.value_or(0.0f),
                    color.r, color.g, color.b, color.a * opacity, kind, 0, 0, 0});
            };
            const auto history = AdsbHistoryDots(ac, context);
            for (U64 i = 0; i < history.size(); ++i) {
                marker(history[i], 2.0f, 1, 0.60f - static_cast<F32>(i) * 0.08f);
            }
            marker(target.ndc, 10, target.heading ? 0 : 2, 1);

            if (!text) continue;
            const auto fill = LeftAlignedBlock(FormatAdsbDataBlock(ac));
            const Extent2D<F32> size{text->advance(fill) * LabelScale + 4 * AdsbTrackingScale,
                static_cast<F32>(text->getConfig().font->lineHeight()) * LabelScale *
                    (1 + static_cast<F32>(std::count(fill.begin(), fill.end(), '\n'))) + 4 * AdsbTrackingScale};
            if (size.x + 12 > viewport.x || size.y + 12 > viewport.y) continue;
            const auto anchor = toPixels(target.ndc);
            const auto placement = PlaceAdsbDataBlock(anchor, size, viewport, occupied, preferredDirections[ac.icao]);
            preferredDirections[ac.icao] = placement.direction;
            const auto& box = placement.box;
            occupied.push_back({box.x - 4 * AdsbTrackingScale, box.y - 4 * AdsbTrackingScale,
                                box.width + 8 * AdsbTrackingScale, box.height + 8 * AdsbTrackingScale});
            const auto end = toNdc({std::clamp(anchor.x, box.x, box.x + box.width),
                                    std::clamp(anchor.y, box.y, box.y + box.height)});
            leaders.append<8>({target.ndc.x, target.ndc.y, end.x, end.y,
                                color.r, color.g, color.b, 0.60f});
            auto element = text->get(ids[slot]);
            element.fill = fill;
            element.scale = LabelScale;
            element.position = toNdc({box.x + 2 * AdsbTrackingScale, box.y + 2 * AdsbTrackingScale});
            element.color = color;
            JST_CHECK(text->update(ids[slot], element));
            element.position.x += pixel.x * AdsbTrackingScale;
            element.position.y -= pixel.y * AdsbTrackingScale;
            element.color = std::nullopt;
            JST_CHECK(shadow->update(ids[slot], element));
            ++slot;
        }
        if (text) {
            for (U64 i = slot; i < previousSlots; ++i) {
                auto element = text->get(ids[i]);
                element.fill.clear();
                element.scale = 0;
                JST_CHECK(text->update(ids[i], element));
                JST_CHECK(shadow->update(ids[i], element));
            }
            previousSlots = slot;
            JST_CHECK(shadow->present());
            JST_CHECK(text->present());
        }
        uniforms = {pixel.x, pixel.y, 2.0f * AdsbTrackingScale, 0};
        JST_CHECK(uniformBuffer->update());
        JST_CHECK(targets.present());
        return leaders.present();
    }

 private:
    std::shared_ptr<const AdsbMapState> state;
    std::unordered_map<U32, U8> preferredDirections;
    std::vector<std::string> ids;
    U64 previousSlots = 0;
    std::array<F32, 4> uniforms{};
    std::shared_ptr<Render::Buffer> quadBuffer, uniformBuffer;
    Batch targets, leaders;
    std::shared_ptr<Text> text, shadow;
};
}  // namespace

Result AddAdsbMapLayers(Render::Components::GeoMap& map,
                       const std::shared_ptr<const AdsbMapState>& state) {
    if (!state) return Result::ERROR;
    return map.addLayer(std::make_shared<RadarLayer>(state));
}

}  // namespace Jetstream::Modules
