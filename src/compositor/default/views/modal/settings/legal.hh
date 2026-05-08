#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_LEGAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_LEGAL_HH

#include "jetstream/render/sakura/sakura.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct LegalSettingsPanel : public Sakura::Component {
    struct Config {
        std::function<void()> onViewFullLicenses;
    };

    void update(Config config) {
        this->config = std::move(config);

        title.update({
            .id = "LegalTitle",
            .str = "Legal",
            .scale = 1.2f,
        });

        description.update({
            .id = "LegalDescription",
            .str = "Software license and third-party acknowledgments.",
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
        });

        divider.update({
            .id = "LegalHeaderDivider",
        });

        tabs.update({
            .id = "##legal-tabs",
            .labels = {"License", "Third-Party Licenses"},
        });

        spacing.update({
            .id = "LegalSpacing",
        });

        licenseScroll.update({
            .id = "license_scroll",
        });

        thirdPartyScroll.update({
            .id = "oss_scroll",
        });

        licenseText.update({
            .id = "LegalLicenseText",
            .str = "MIT License\n"
                   "\n"
                   "Copyright (c) 2021-2026 Luigi F. Cruz\n"
                   "\n"
                   "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
                   "of this software and associated documentation files (the \"Software\"), to deal\n"
                   "in the Software without restriction, including without limitation the rights\n"
                   "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
                   "copies of the Software, and to permit persons to whom the Software is\n"
                   "furnished to do so, subject to the following conditions:\n"
                   "\n"
                   "The above copyright notice and this permission notice shall be\n"
                   "included in all copies or substantial portions of the Software.\n"
                   "\n"
                   "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
                   "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
                   "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
                   "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
                   "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
                   "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
                   "SOFTWARE.",
        });

        introText.update({
            .id = "LegalIntroText",
            .str = "CyberEther utilizes the following open-source third-party software,\n"
                   "and we extend our gratitude to the creators of these libraries for\n"
                   "their valuable contributions to the open-source community.",
        });

        thirdPartyText.update({
            .id = "LegalThirdPartyText",
            .str = "- Miniaudio - MIT License\n"
                   "- Dear ImGui - MIT License\n"
                   "- ImNodes - MIT License\n"
                   "- PocketFFT - BSD-3-Clause License\n"
                   "- RapidYAML - MIT License\n"
                   "- vkFFT - MIT License\n"
                   "- stb - MIT License\n"
                   "- fmtlib - MIT License\n"
                   "- SoapySDR - Boost Software License\n"
                   "- libmodes - BSD-2-Clause License\n"
                   "- GLFW - zlib/libpng License\n"
                   "- imgui-notify - MIT License\n"
                   "- spirv-cross - MIT License\n"
                   "- glslang - BSD-3-Clause License\n"
                   "- naga - Apache License 2.0\n"
                   "- gstreamer - LGPL-2.1 License\n"
                   "- libusb - LGPL-2.1 License\n"
                   "- Nanobench - MIT License\n"
                   "- Catch2 - Boost Software License\n"
                   "- JetBrains Mono - SIL Open Font License 1.1\n"
                   "- imgui_markdown - Zlib License\n"
                   "- GLM - Happy Bunny License\n"
                   "- cpp-httplib - MIT License\n"
                   "- nlohmann/json - MIT License\n"
                   "- Natural Earth - Public Domain\n"
                   // [NEW DEPENDENCY HOOK]
        });

        viewFullLicensesButton.update({
            .id = "LegalViewFullLicenses",
            .str = "View Full Licenses",
            .size = {-1.0f, 0.0f},
            .onClick = [this]() {
                if (this->config.onViewFullLicenses) {
                    this->config.onViewFullLicenses();
                }
            },
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        description.render(ctx);
        divider.render(ctx);

        tabs.render(ctx, {
            [this](const Sakura::Context& ctx) {
                spacing.render(ctx);
                licenseScroll.render(ctx, [&](const Sakura::Context& ctx) {
                    licenseText.render(ctx);
                });
            },
            [this](const Sakura::Context& ctx) {
                spacing.render(ctx);
                introText.render(ctx);
                spacing.render(ctx);

                thirdPartyScroll.render(ctx, [&](const Sakura::Context& ctx) {
                    thirdPartyText.render(ctx);
                });

                spacing.render(ctx);
                viewFullLicensesButton.render(ctx);
            },
        });
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::TabBar tabs;
    Sakura::Spacing spacing;
    Sakura::ScrollArea licenseScroll;
    Sakura::ScrollArea thirdPartyScroll;
    Sakura::Button viewFullLicensesButton;
    Sakura::Text licenseText;
    Sakura::Text introText;
    Sakura::Text thirdPartyText;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_LEGAL_HH
