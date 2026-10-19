#include <catch2/catch_test_macros.hpp>

#include "component_harness.hh"
#include "render/sakura/detail/drawable.hh"
#include "render/sakura/detail/root_component.hh"

#include <memory>
#include <unordered_map>
#include <vector>

using namespace Jetstream;

namespace {

struct PopupContent : Sakura::Component {
    Extent2D<F32> measure(const Sakura::Context&, Extent2D<F32>) override { return {100, 30}; }
};

struct PopupRequester : Sakura::Component {
    std::vector<Sakura::Component*> shown;

    void show(std::vector<Sakura::Component*> next) {
        shown = std::move(next);
        invalidate(Dirty::Layout);
    }

    void layout(const Sakura::Context& ctx) override {
        for (auto* content : shown) {
            Sakura::RequestPopup(ctx, {.content = content, .anchor = {100, 100, 20, 20}, .panel = true});
        }
    }
};

struct DrawableContext : Sakura::Drawable {
    static void set(Sakura::Drawable& drawable, Sakura::Context& ctx) {
        drawable.*(&DrawableContext::context) = &ctx;
    }
};

struct PopupLifetimeFixture {
    std::unordered_map<Sakura::Drawable*, std::shared_ptr<Render::Program>> programs;
    std::unordered_map<const Sakura::Component*, Sakura::Drawable*> owners;
    Sakura::Context ctx;
    PopupRequester requester;
    Sakura::Layer layer;
    Sakura::RootComponent root;

    PopupLifetimeFixture() {
        ctx.pixelRatio = 1;
        ctx.framebufferSize = {800, 600};
        ctx.attach = [&](const Sakura::Component& owner, Sakura::Drawable* drawable) {
            DrawableContext::set(*drawable, ctx);
            owners[&owner] = drawable;
            if (!programs.contains(drawable)) {
                programs[drawable] = std::make_shared<Render::Program>(Render::Program::Config{});
            }
            return Result::SUCCESS;
        };
        ctx.release = [&](Sakura::Drawable* drawable, std::shared_ptr<Render::Components::Generic>) {
            programs.erase(drawable);
            std::erase_if(owners, [&](const auto& entry) { return entry.second == drawable; });
        };
        layer.update({.content = &requester});
        root.setLayers({&layer});
    }

    void frame() {
        ++ctx.frame;
        Sakura::Access::layout(root, ctx, {0, 0, 800, 600});
        Sakura::Access::present(root, ctx.frame);
        REQUIRE(Sakura::Access::build(root, ctx) == Result::SUCCESS);
    }

    std::vector<Sakura::Box*> frames() const {
        std::vector<Sakura::Box*> result;
        for (auto* child : SakuraTest::Peek::Children(root.popups)) {
            if (auto* box = dynamic_cast<Sakura::Box*>(child)) result.push_back(box);
        }
        return result;
    }
};

}  // namespace

TEST_CASE("Popup backgrounds reclaim dead contents and release their attachment programs without disturbing open caches",
          "[core][sakura][popup-lifetime]") {
    PopupLifetimeFixture fixture;
    PopupContent persistent;
    fixture.requester.show({&persistent});
    fixture.frame();
    REQUIRE(fixture.frames().size() == 1);
    auto* permanentFrame = fixture.frames().front();
    const auto permanentAlive = Sakura::Access::alive(*permanentFrame);
    const std::weak_ptr<Render::Program> permanentProgram = fixture.programs.at(fixture.owners.at(permanentFrame));
    const U64 basePrograms = fixture.programs.size();

    auto cached = std::make_unique<PopupContent>();
    fixture.requester.show({&persistent, cached.get()});
    fixture.frame();
    REQUIRE(fixture.frames().size() == 2);
    std::weak_ptr<bool> cachedFrame;
    for (auto* frame : fixture.frames()) {
        if (frame != permanentFrame) cachedFrame = Sakura::Access::alive(*frame);
    }
    fixture.requester.show({&persistent});
    fixture.frame();
    CHECK_FALSE(cachedFrame.expired());
    fixture.requester.show({&persistent, cached.get()});
    fixture.frame();
    CHECK_FALSE(cachedFrame.expired());
    fixture.requester.show({&persistent});
    cached.reset();
    fixture.frame();
    CHECK(cachedFrame.expired());
    CHECK(fixture.frames().size() == 1);
    CHECK(fixture.programs.size() == basePrograms);

    for (U64 batch = 0; batch < 8; ++batch) {
        std::vector<std::unique_ptr<PopupContent>> transient;
        std::vector<Sakura::Component*> shown{&persistent};
        for (U64 index = 0; index < 16; ++index) {
            transient.push_back(std::make_unique<PopupContent>());
            shown.push_back(transient.back().get());
        }
        fixture.requester.show(shown);
        fixture.frame();
        REQUIRE(fixture.frames().size() == 17);
        CHECK(fixture.programs.size() == basePrograms + 16);
        fixture.requester.show({&persistent});
        transient.clear();
        fixture.frame();
        CHECK(fixture.frames().size() == 1);
        CHECK(fixture.programs.size() == basePrograms);
        CHECK_FALSE(permanentAlive.expired());
        CHECK_FALSE(permanentProgram.expired());
        CHECK(fixture.root.popups.interactiveAt({110, 125}));
    }
}
