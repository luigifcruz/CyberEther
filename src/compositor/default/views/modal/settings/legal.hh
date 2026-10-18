#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_LEGAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_LEGAL_HH

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <string>

namespace Jetstream {

struct LegalSettingsPanel {
    struct Config {
        std::function<void()> onViewFullLicenses;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        this->config = std::move(config);
        title.update({
            .str = "Legal",
            .font = Text::Font::Bold,
            .scale = 1.2f,
        });
        description.update({
            .str = "Software license and third-party acknowledgments.",
            .tone = Text::Tone::Secondary,
            .wrapped = true,
        });
        divider.update({});
        licenseSpacing.update({});
        introSpacing.update({});
        scrollSpacing.update({});
        buttonSpacing.update({});
        licenseText.update({
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
            .str = "CyberEther utilizes the following open-source third-party software,\n"
                   "and we extend our gratitude to the creators of these libraries for\n"
                   "their valuable contributions to the open-source community.",
        });
        thirdPartyText.update({
            .str = "- Miniaudio - MIT License\n"
                   "- Dear ImGui - MIT License\n"
                   "- PocketFFT - BSD-3-Clause License\n"
                   "- RapidYAML - MIT License\n"
                   "- toml++ - MIT License\n"
                   "- vkFFT - MIT License\n"
                   "- stb - MIT License\n"
                   "- fmtlib - MIT License\n"
                   "- SoapySDR - Boost Software License\n"
                   "- libhackrf - BSD-3-Clause License\n"
                   "- SoapyHackRF - MIT License\n"
                   "- libbladeRF - LGPL-2.1-or-later License\n"
                   "- SoapyBladeRF - LGPL-2.1-or-later License\n"
                   "- Analog Devices no-OS - Analog Devices BSD-style License\n"
                   "- libmodes - BSD-2-Clause License\n"
                   "- tree-sitter - MIT License\n"
                   "- tree-sitter-python - MIT License\n"
                   "- tree-sitter-markdown - MIT License\n"
                   "- tree-sitter-yaml - MIT License\n"
                   "- tree-sitter-bash - MIT License\n"
                   "- tree-sitter-cpp - MIT License\n"
                   "- tree-sitter-json - MIT License\n"
                   "- ONNX Runtime - MIT License\n"
                   "- GLFW - zlib/libpng License\n"
                   "- spirv-cross - Apache License 2.0\n"
                   "- Apple metal-cpp - Apache License 2.0\n"
                   "- glslang - BSD-3-Clause License\n"
                   "- naga - Apache License 2.0\n"
                   "- gstreamer - LGPL-2.1 License\n"
                   "- PCRE2 - BSD-3-Clause WITH PCRE2-exception License\n"
                   "- libffi - MIT License\n"
                   "- GLib - LGPL-2.1-or-later License\n"
                   "- libsrtp2 - BSD-3-Clause License\n"
                   "- libvpx - BSD-3-Clause License\n"
                   "- libnice - LGPL-2.1-or-later License\n"
                   "- libgudev - LGPL-2.1-or-later License\n"
                   "- gl-headers - MIT/Apache-2.0 License\n"
                   "- libusb - LGPL-2.1 License\n"
                   "- libqrencode - LGPL-2.1-or-later License\n"
                   "- zlib - zlib License\n"
                   "- OpenSSL - Apache License 2.0\n"
                   "- Nanobench - MIT License\n"
                   "- Catch2 - Boost Software License\n"
                   "- Font Awesome - Font Awesome Free License\n"
                   "- JetBrains Mono - SIL Open Font License 1.1\n"
                   "- Space Grotesk - SIL Open Font License 1.1\n"
                   "- Inter - SIL Open Font License 1.1\n"
                   "- GLM - Happy Bunny/MIT License\n"
                   "- cpp-httplib - MIT License\n"
                   "- nlohmann/json - MIT License\n"
                   "- nanobind - BSD 3-Clause License\n"
                   "- robin-map - MIT License\n"
                   "- Google Turbo colormap - Apache License 2.0\n"
                   "- Natural Earth - Public Domain\n"
                   "- Velopack - MIT License\n"
                   // [NEW DEPENDENCY HOOK]
        });
        viewFullLicensesButton.update({
            .str = "View Full Licenses",
            .size = {Sakura::Fill, 0.0f},
            .onClick = [this]() {
                if (this->config.onViewFullLicenses) {
                    this->config.onViewFullLicenses();
                }
            },
        });
        licenseScroll.update({.content = &licenseText});
        thirdPartyScroll.update({.content = &thirdPartyText});
        licensePage.update({
            .items = {
                {.child = &licenseSpacing},
                {.child = &licenseScroll},
            },
        });
        thirdPartyPage.update({
            .items = {
                {.child = &introSpacing},
                {.child = &introText},
                {.child = &scrollSpacing},
                {.child = &thirdPartyScroll},
                {.child = &buttonSpacing},
                {.child = &viewFullLicensesButton},
            },
        });
        tabs.update({
            .labels = {"License", "Third-Party Licenses"},
            .pages = {&licensePage, &thirdPartyPage},
        });
        stack.update({
            .items = {
                {.child = &title},
                {.child = &description},
                {.child = &divider},
                {.child = &tabs},
            },
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Text description;
    Sakura::Divider divider;
    Sakura::Spacer licenseSpacing;
    Sakura::Spacer introSpacing;
    Sakura::Spacer scrollSpacing;
    Sakura::Spacer buttonSpacing;
    Sakura::Text licenseText;
    Sakura::Text introText;
    Sakura::Text thirdPartyText;
    Sakura::Button viewFullLicensesButton;
    Sakura::ScrollArea licenseScroll;
    Sakura::ScrollArea thirdPartyScroll;
    Sakura::Stack licensePage;
    Sakura::Stack thirdPartyPage;
    Sakura::TabBar tabs;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_SETTINGS_LEGAL_HH
