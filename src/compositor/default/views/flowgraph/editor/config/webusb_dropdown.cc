#include "webusb_dropdown.hh"

#include "render/sakura/context.hh"

#ifdef JST_OS_BROWSER
#include <emscripten.h>
#endif

namespace Jetstream {

Result FlowgraphConfigWebUsbDropdownField::update(Config config) {
    this->config = std::move(config);
    JST_CHECK(dropdown.update(this->config));
    authorizeButton.update({
        .str = "Authorize SDR Device",
        .size = {Sakura::Fill, 0.0f},
        .variant = Sakura::Button::Variant::Action,
        .onClick = [this]() {
#ifdef JST_OS_BROWSER
            MAIN_THREAD_EM_ASM({
                if (!navigator.usb) {
                    console.error("This browser does not support WebUSB.");
                    return;
                }
                if (Module['sdrUsbRequestPending']) {
                    return;
                }
                Module['sdrUsbRequestPending'] = true;
                navigator.usb.requestDevice({
                    filters: [{vendorId: 0x0bda}, {vendorId: 0x1d50}, {classCode: 0xff}],
                }).then(() => {
                    Module['sdrUsbAuthorizationGeneration'] =
                        ((Module['sdrUsbAuthorizationGeneration'] || 0) + 1) & 0x7fffffff;
                }).catch(error => {
                    if (error.name !== "NotFoundError") {
                        console.error("WebUSB authorization failed:", error);
                    }
                }).finally(() => {
                    Module['sdrUsbRequestPending'] = false;
                });
            });
            requestFrame();
#else
            if (this->config.onError) {
                this->config.onError(Result::ERROR, "WebUSB authorization requires a browser build.");
            }
#endif
        },
    });
    Sakura::Stack::update({
        .items = {{.child = &dropdown.component()}, {.child = &authorizeButton}},
    });
    return Result::SUCCESS;
}

Sakura::Component& FlowgraphConfigWebUsbDropdownField::component() {
    return *this;
}

void FlowgraphConfigWebUsbDropdownField::layout(const Sakura::Context& ctx) {
    Sakura::Stack::layout(ctx);
#ifdef JST_OS_BROWSER
    const int generation = MAIN_THREAD_EM_ASM_INT({
        return Module['sdrUsbAuthorizationGeneration'] || 0;
    });
    if (observedGeneration < 0) {
        observedGeneration = generation;
    } else if (generation != observedGeneration) {
        observedGeneration = generation;
        U64 refresh = 0;
        if (Parser::Deserialize(config.values, "webUsbRefresh", refresh) != Result::SUCCESS) {
            if (config.onError) {
                config.onError(Result::ERROR, "Invalid WebUSB refresh value.");
            }
        } else if (config.onApply) {
            config.onApply({{"webUsbRefresh", refresh + 1}}, false, {});
        }
    }
    requestFrame(ctx.time + 0.25);
#endif
}

}  // namespace Jetstream
