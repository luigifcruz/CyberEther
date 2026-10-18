#include <catch2/catch_test_macros.hpp>

#include "compositor/default/actions/flowgraph.hh"
#include "compositor/default/actions/workbench.hh"
#include "compositor/default/model/surface_theme.hh"
#include "compositor/default/themes.hh"

#include "jetstream/platform.hh"
#include "jetstream/settings.hh"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace Jetstream;

namespace {

class SettingsSandbox {
 public:
    SettingsSandbox() {
        static std::atomic<U64> sequence{0};
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        root = std::filesystem::temp_directory_path() /
               Platform::PathFromUtf8("cyberether-compositor-theme-" + std::to_string(nonce) + "-" +
                                      std::to_string(sequence.fetch_add(1)));
#if defined(JST_OS_LINUX)
        redirected = redirect("HOME", root / "home") && redirect("XDG_CONFIG_HOME", root / "config");
#elif defined(JST_OS_MAC)
        redirected = redirect("CFFIXED_USER_HOME", root);
#endif
    }

    ~SettingsSandbox() {
        for (auto it = saved.rbegin(); it != saved.rend(); ++it) {
            if (it->second) {
                (void)setenv(it->first.c_str(), it->second->c_str(), 1);
            } else {
                (void)unsetenv(it->first.c_str());
            }
        }
        std::error_code ec;
        std::filesystem::remove_all(root, ec);
    }

    std::optional<std::filesystem::path> settingsPath() const {
        std::string configPath;
        if (!redirected || Platform::ConfigPath(configPath) != Result::SUCCESS) {
            return std::nullopt;
        }
        const auto path = Platform::PathFromUtf8(configPath) / "settings.yaml";
        const auto relative = path.lexically_relative(root);
        if (relative.empty() || *relative.begin() == "..") {
            return std::nullopt;
        }
        return path;
    }

 private:
    bool redirect(const char* name, const std::filesystem::path& value) {
        std::string current;
        saved.emplace_back(name, Platform::EnvironmentVariable(name, current) == Result::SUCCESS
                                     ? std::optional<std::string>(current)
                                     : std::nullopt);
        return setenv(name, Platform::PathToUtf8(value).c_str(), 1) == 0;
    }

    std::filesystem::path root;
    bool redirected = false;
    std::vector<std::pair<std::string, std::optional<std::string>>> saved;
};

struct TrackedSurfaces {
    std::shared_ptr<Module::Surface> attached = std::make_shared<Module::Surface>();
    std::shared_ptr<Module::Surface> detached = std::make_shared<Module::Surface>();
};

TrackedSurfaces TrackSurfaces(DefaultCompositorState& state, DefaultCompositorCallbacks& callbacks) {
    TrackedSurfaces surfaces;
    FlowgraphActions actions(state, callbacks);
    REQUIRE(actions.handle(MailResizeSurface{
        .surface = surfaces.attached,
        .placement = SurfacePlacement::Attached,
        .resize = {.framebufferSize = {320, 200}, .scale = 1.0f},
    }) == Result::SUCCESS);
    REQUIRE(actions.handle(MailResizeSurface{
        .surface = surfaces.detached,
        .placement = SurfacePlacement::Detached,
        .resize = {.framebufferSize = {640, 480}, .scale = 2.0f},
    }) == Result::SUCCESS);
    REQUIRE(state.flowgraph.surfaces.size() == 2);
    return surfaces;
}

}  // namespace

TEST_CASE("Applying a theme rethemes tracked surfaces in place and saves the theme key",
          "[core][compositor][theme]") {
    SettingsSandbox sandbox;
    const auto settingsPath = sandbox.settingsPath();
    if (!settingsPath) {
        SKIP("Settings cannot be redirected away from the user configuration on this platform.");
    }
    REQUIRE(themes.contains("Dark"));
    REQUIRE(themes.contains("Light"));

    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<std::string> notifications;
    callbacks.notify = [&](Sakura::ToastType, I32, const std::string& message) {
        notifications.push_back(message);
    };
    state.sakura.themeKey = "Dark";
    state.sakura.colorMap = themes.at("Dark");
    const auto surfaces = TrackSurfaces(state, callbacks);
    WorkbenchActions actions(state, callbacks);

    REQUIRE(actions.handle(MailApplyTheme{"Light"}) == Result::SUCCESS);
    CHECK(notifications.empty());
    CHECK(state.sakura.themeKey == "Light");
    CHECK(state.sakura.colorMap == themes.at("Light"));
    CHECK(&state.sakura.runtime.context().palette.get() == &state.sakura.colorMap);

    const auto& light = themes.at("Light");
    const auto& attached = state.flowgraph.surfaces.at(surfaces.attached.get()).event;
    CHECK(attached.placement == SurfacePlacementType::Attached);
    CHECK(attached.theme == MakeSurfaceTheme(light, SurfacePlacementType::Attached));
    CHECK(attached.size == Extent2D<U64>{320, 200});
    CHECK(attached.scale == 1.0f);
    const auto& detached = state.flowgraph.surfaces.at(surfaces.detached.get()).event;
    CHECK(detached.placement == SurfacePlacementType::Detached);
    CHECK(detached.theme == MakeSurfaceTheme(light, SurfacePlacementType::Detached));
    CHECK(detached.size == Extent2D<U64>{640, 480});
    CHECK(detached.scale == 2.0f);

    Settings settings;
    REQUIRE(Settings::Get(settings) == Result::SUCCESS);
    CHECK(settings.interface.themeKey == "Light");
    CHECK(std::filesystem::exists(*settingsPath));
}

TEST_CASE("Applying an unknown theme notifies and leaves surfaces and settings untouched",
          "[core][compositor][theme]") {
    SettingsSandbox sandbox;
    const auto settingsPath = sandbox.settingsPath();
    if (!settingsPath) {
        SKIP("Settings cannot be redirected away from the user configuration on this platform.");
    }
    REQUIRE(themes.contains("Dark"));

    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::vector<std::pair<Sakura::ToastType, std::string>> notifications;
    callbacks.notify = [&](Sakura::ToastType type, I32, const std::string& message) {
        notifications.emplace_back(type, message);
    };
    state.sakura.themeKey = "Dark";
    state.sakura.colorMap = themes.at("Dark");
    const auto surfaces = TrackSurfaces(state, callbacks);
    const auto attachedTheme = state.flowgraph.surfaces.at(surfaces.attached.get()).event.theme;
    const auto detachedTheme = state.flowgraph.surfaces.at(surfaces.detached.get()).event.theme;
    WorkbenchActions actions(state, callbacks);

    REQUIRE(actions.handle(MailApplyTheme{"Missing"}) == Result::SUCCESS);
    REQUIRE(notifications.size() == 1);
    CHECK(notifications.front().first == Sakura::ToastType::Error);
    CHECK_FALSE(notifications.front().second.empty());
    CHECK(state.sakura.themeKey == "Dark");
    CHECK(state.sakura.colorMap == themes.at("Dark"));
    CHECK(state.flowgraph.surfaces.at(surfaces.attached.get()).event.theme == attachedTheme);
    CHECK(state.flowgraph.surfaces.at(surfaces.detached.get()).event.theme == detachedTheme);
    CHECK_FALSE(std::filesystem::exists(*settingsPath));
}
