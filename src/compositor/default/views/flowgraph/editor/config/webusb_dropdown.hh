#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_WEBUSB_DROPDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_WEBUSB_DROPDOWN_HH

#include "dropdown.hh"

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#endif

#include <any>
#include <limits>

namespace Jetstream {

struct FlowgraphConfigWebUsbDropdownField {
    using Config = FlowgraphConfigFieldConfig;

    void update(Config config) {
        this->config = std::move(config);

        auto dropdownConfig = this->config;
        const auto separator = dropdownConfig.format.find(':');
        dropdownConfig.format = "dropdown";
        if (separator != std::string::npos) {
            dropdownConfig.format += this->config.format.substr(separator);
        }
        dropdown.update(std::move(dropdownConfig));

        authorizeButton.update({
            .id = this->config.id + "AuthorizeUsb",
            .str = "Authorize SDR Device",
            .size = {-1.0f, 0.0f},
            .variant = Sakura::Button::Variant::Action,
            .onClick = []() {
#ifdef JST_OS_BROWSER
                MAIN_THREAD_EM_ASM({
                    if (!navigator.usb || globalThis.cyberetherWebUsbRequestPending) {
                        return;
                    }
                    globalThis.cyberetherWebUsbRequestPending = true;
                    navigator.usb.requestDevice({
                        filters: [
                            {vendorId: 0x0bda},
                            {vendorId: 0x1d50},
                            {classCode: 0xff},
                        ],
                    }).then(() => {
                        globalThis.cyberetherWebUsbGeneration =
                            (globalThis.cyberetherWebUsbGeneration || 0) + 1;
                    }).catch(error => {
                        if (error.name !== "NotFoundError") {
                            console.error("WebUSB authorization failed:", error);
                        }
                    }).finally(() => {
                        globalThis.cyberetherWebUsbRequestPending = false;
                    });
                });
#endif
            },
        });

#ifdef JST_OS_BROWSER
        if (!generationInitialized) {
            observedGeneration = CurrentGeneration();
            generationInitialized = true;
        }
#endif
    }

    void render(const Sakura::Context& ctx) const {
        dropdown.render(ctx);
        authorizeButton.render(ctx);

#ifdef JST_OS_BROWSER
        const U64 generation = CurrentGeneration();
        if (generation == observedGeneration) {
            return;
        }
        observedGeneration = generation;

        U64 refresh = 0;
        if (config.values.contains("webUsbRefresh")) {
            try {
                refresh = std::any_cast<U64>(config.values.at("webUsbRefresh"));
            } catch (const std::bad_any_cast&) {
            }
        }
        if (refresh == std::numeric_limits<U64>::max()) {
            refresh = 0;
        } else {
            ++refresh;
        }

        Parser::Map patch;
        patch["webUsbRefresh"] = refresh;
        if (config.onApply) {
            config.onApply(std::move(patch), false);
        }
#endif
    }

 private:
#ifdef JST_OS_BROWSER
    static U64 CurrentGeneration() {
        return static_cast<U64>(MAIN_THREAD_EM_ASM_INT({
            return globalThis.cyberetherWebUsbGeneration || 0;
        }));
    }
#endif

    Config config;
    FlowgraphConfigDropdownField dropdown;
    mutable Sakura::Button authorizeButton;
#ifdef JST_OS_BROWSER
    mutable U64 observedGeneration = 0;
    bool generationInitialized = false;
#endif
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_WEBUSB_DROPDOWN_HH
