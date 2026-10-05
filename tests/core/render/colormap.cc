#include <catch2/catch_test_macros.hpp>

#include <vector>

#include <jetstream/render/colormap.hh>

#if defined(JETSTREAM_RENDER_METAL_AVAILABLE)
#include <jetstream/render/devices/metal/window.hh>
#elif defined(JETSTREAM_RENDER_VULKAN_AVAILABLE)
#include <jetstream/render/devices/vulkan/window.hh>
#elif defined(JETSTREAM_RENDER_WEBGPU_AVAILABLE)
#include <jetstream/render/devices/webgpu/window.hh>
#endif

using namespace Jetstream;

TEST_CASE("Standard palettes are supported and removed palettes are rejected",
          "[core][render][colormap]") {
    for (const auto* name : {"grayscale", "turbo", "viridis", "inferno",
                             "magma", "plasma", "jet"}) {
        CAPTURE(name);
        REQUIRE(Render::Colormap::Valid(name));
    }
    REQUIRE_FALSE(Render::Colormap::Valid("unknown"));
    REQUIRE_FALSE(Render::Colormap::Valid("amber"));
    REQUIRE_FALSE(Render::Colormap::Valid("copper"));
    REQUIRE_FALSE(Render::Colormap::Valid("parula"));
}

TEST_CASE("The shared colormap dropdown exposes Jet without removed palettes",
          "[core][render][colormap]") {
    const auto format = Render::Colormap::Format();
    REQUIRE(Parser::Get<std::string>(format, "type") == "dropdown");

    const auto options = Parser::Get<std::vector<Parser::Map>>(format, "options");
    U64 matchingOptions = 0;
    for (const auto& option : options) {
        const auto value = Parser::Get<std::string>(option, "value");
        REQUIRE(Render::Colormap::Valid(value));
        REQUIRE(value != "amber");
        REQUIRE(value != "copper");
        REQUIRE(value != "parula");
        if (value == "jet") {
            REQUIRE(Parser::Get<std::string>(option, "label") == "Jet");
            ++matchingOptions;
        }
    }
    REQUIRE(matchingOptions == 1);
}

TEST_CASE("Jet LUT has expected colors and supports palette updates",
          "[core][render][colormap]") {
    std::shared_ptr<Render::Window> window;
#if defined(JETSTREAM_RENDER_METAL_AVAILABLE)
    window = std::make_shared<Render::WindowImp<DeviceType::Metal>>(Render::Window::Config{},
                                                                 nullptr);
#elif defined(JETSTREAM_RENDER_VULKAN_AVAILABLE)
    window = std::make_shared<Render::WindowImp<DeviceType::Vulkan>>(Render::Window::Config{},
                                                                  nullptr);
#elif defined(JETSTREAM_RENDER_WEBGPU_AVAILABLE)
    window = std::make_shared<Render::WindowImp<DeviceType::WebGPU>>(Render::Window::Config{},
                                                                  nullptr);
#else
    SKIP("No rendering backend is enabled.");
#endif

    // Building the texture descriptor does not initialize a window or GPU resources.
    Render::Colormap colormap;
    REQUIRE(colormap.create(window, "jet") == Result::SUCCESS);
    REQUIRE(colormap.texture());
    const auto& config = colormap.texture()->getConfig();
    REQUIRE(config.size == Extent2D<U64>{256, 1});
    REQUIRE(config.buffer);

    const auto color = [&](U64 index) {
        const auto* pixel = config.buffer + index * 4;
        return std::array<U8, 4>{pixel[0], pixel[1], pixel[2], pixel[3]};
    };
    const std::array<U8, 4> midpoint = {130, 255, 126, 255};
    REQUIRE(color(0) == std::array<U8, 4>{0, 0, 128, 255});
    REQUIRE(color(32) == std::array<U8, 4>{0, 1, 255, 255});
    REQUIRE(color(64) == std::array<U8, 4>{0, 129, 255, 255});
    REQUIRE(color(96) == std::array<U8, 4>{2, 255, 254, 255});
    REQUIRE(color(160) == std::array<U8, 4>{255, 253, 0, 255});
    REQUIRE(color(192) == std::array<U8, 4>{255, 125, 0, 255});
    REQUIRE(color(224) == std::array<U8, 4>{252, 0, 0, 255});
    REQUIRE(color(255) == std::array<U8, 4>{128, 0, 0, 255});
    REQUIRE(color(128) == midpoint);
    REQUIRE(color(0)[3] == 255);

    for (U64 index = 1; index < 256; ++index) {
        const auto previous = color(index - 1);
        const auto current = color(index);
        CAPTURE(index);
        for (U64 channel = 0; channel < 3; ++channel) {
            const I32 difference = static_cast<I32>(current[channel]) -
                                   static_cast<I32>(previous[channel]);
            REQUIRE(difference >= -4);
            REQUIRE(difference <= 4);
        }
        REQUIRE(current[3] == 255);
    }

    REQUIRE(colormap.update("grayscale") == Result::SUCCESS);
    REQUIRE(color(128) == std::array<U8, 4>{128, 128, 128, 255});
    REQUIRE(colormap.update("jet") == Result::SUCCESS);
    REQUIRE(color(128) == midpoint);
}
