#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_LEGAL_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_LEGAL_HH

#include "components/settings_group.hh"
#include "components/settings_header.hh"

#include "jetstream/render/sakura/base.hh"

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct LegalSettingsPanel {
    struct Config {
        std::function<void()> onViewFullLicenses;
    };

    void update(Config config) {
        this->config = std::move(config);
        header.update({
            .title = "Legal",
            .description = "Software license and third-party acknowledgments.",
        });
        licenseGroup.update({
            .title = "License",
            .description = "CyberEther is released under the MIT License.",
            .rows = {
                {
                    .label = "Copyright",
                    .description = "Copyright (c) 2021-2026 Luigi F. Cruz",
                },
                {
                    .label = "Permission",
                    .description = "Permission is hereby granted, free of charge, to any person obtaining a copy of this "
                                   "software and associated documentation files (the \"Software\"), to deal in the "
                                   "Software without restriction, including without limitation the rights to use, copy, "
                                   "modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, "
                                   "and to permit persons to whom the Software is furnished to do so, subject to the "
                                   "following conditions.",
                },
                {
                    .label = "Conditions",
                    .description = "The above copyright notice and this permission notice shall be included in all "
                                   "copies or substantial portions of the Software.",
                },
                {
                    .label = "Warranty",
                    .description = "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, "
                                   "INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A "
                                   "PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT "
                                   "HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF "
                                   "CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE "
                                   "OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.",
                },
            },
        });

        const auto& dependencies = Dependencies();
        while (licenses.size() < dependencies.size()) {
            licenses.push_back(std::make_unique<Sakura::Badge>());
        }
        std::vector<Sakura::SettingCard::Row> rows;
        rows.reserve(dependencies.size());
        for (U64 index = 0; index < dependencies.size(); ++index) {
            licenses[index]->update({
                .label = dependencies[index].license,
                .backgroundColorKey = "header_hovered",
                .borderColorKey = "border",
                .textColorKey = "text_secondary",
            });
            rows.push_back({
                .label = dependencies[index].name,
                .control = licenses[index].get(),
                .wide = true,
            });
        }
        fullLicensesButton.update({
            .str = "View Full Licenses",
            .textScale = 0.9f,
            .padding = {9.0f, 5.0f},
            .onClick = [this]() {
                if (this->config.onViewFullLicenses) {
                    this->config.onViewFullLicenses();
                }
            },
            .font = Sakura::Text::Font::Sans,
        });
        thirdPartyGroup.update({
            .title = "Third-Party Software",
            .description = "Open-source projects CyberEther builds on, with thanks to their creators.",
            .rows = std::move(rows),
            .action = &fullLicensesButton,
        });

        stack.update({
            .spacing = 4.0f,
            .items = {
                {.child = &header},
                {.child = &licenseGroup},
                {.child = &thirdPartyGroup},
            },
        });
    }

    Sakura::Component& content() {
        return stack;
    }

 private:
    struct Dependency {
        std::string name;
        std::string license;
    };

    static const std::vector<Dependency>& Dependencies() {
        static const std::vector<Dependency> dependencies = {
            {"Miniaudio", "MIT"},
            {"PocketFFT", "BSD-3-Clause"},
            {"RapidYAML", "MIT"},
            {"toml++", "MIT"},
            {"vkFFT", "MIT"},
            {"stb", "MIT"},
            {"fmtlib", "MIT"},
            {"librtlsdr", "GPL-2.0-or-later"},
            {"libairspy", "BSD-3-Clause"},
            {"libairspyhf", "BSD-3-Clause"},
            {"libhackrf", "BSD-3-Clause"},
            {"LimeSuite", "Apache License 2.0"},
            {"libbladeRF", "LGPL-2.1-or-later"},
            {"Analog Devices no-OS", "Analog Devices BSD-style License"},
            {"libmodes", "BSD-2-Clause"},
            {"tree-sitter", "MIT"},
            {"tree-sitter-python", "MIT"},
            {"tree-sitter-markdown", "MIT"},
            {"tree-sitter-yaml", "MIT"},
            {"tree-sitter-bash", "MIT"},
            {"tree-sitter-cpp", "MIT"},
            {"tree-sitter-json", "MIT"},
            {"ONNX Runtime", "MIT"},
            {"GLFW", "zlib/libpng"},
            {"spirv-cross", "Apache License 2.0"},
            {"Apple metal-cpp", "Apache License 2.0"},
            {"glslang", "BSD-3-Clause"},
            {"naga", "Apache License 2.0"},
            {"gstreamer", "LGPL-2.1"},
            {"PCRE2", "BSD-3-Clause WITH PCRE2-exception"},
            {"libffi", "MIT"},
            {"GLib", "LGPL-2.1-or-later"},
            {"libsrtp2", "BSD-3-Clause"},
            {"libvpx", "BSD-3-Clause"},
            {"libnice", "LGPL-2.1-or-later"},
            {"libgudev", "LGPL-2.1-or-later"},
            {"gl-headers", "MIT/Apache-2.0"},
            {"libusb", "LGPL-2.1"},
            {"libqrencode", "LGPL-2.1-or-later"},
            {"zlib", "zlib License"},
            {"OpenSSL", "Apache License 2.0"},
            {"Nanobench", "MIT"},
            {"Catch2", "Boost Software License"},
            {"Lucide", "ISC License"},
            {"Font Awesome", "Font Awesome Free License"},
            {"JetBrains Mono", "SIL Open Font License 1.1"},
            {"Space Grotesk", "SIL Open Font License 1.1"},
            {"Inter", "SIL Open Font License 1.1"},
            {"GLM", "Happy Bunny/MIT"},
            {"cpp-httplib", "MIT"},
            {"nlohmann/json", "MIT"},
            {"nanobind", "BSD 3-Clause"},
            {"robin-map", "MIT"},
            {"Google Turbo colormap", "Apache License 2.0"},
            {"Natural Earth", "Public Domain"},
            {"Velopack", "MIT"},
            // [NEW DEPENDENCY HOOK]
        };
        return dependencies;
    }

    Config config;
    SettingsHeader header;
    SettingsGroup licenseGroup;
    std::vector<std::unique_ptr<Sakura::Badge>> licenses;
    Sakura::Button fullLicensesButton;
    SettingsGroup thirdPartyGroup;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_SETTINGS_LEGAL_HH
