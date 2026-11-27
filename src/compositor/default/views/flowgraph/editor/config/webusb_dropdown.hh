#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_WEBUSB_DROPDOWN_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CONFIG_WEBUSB_DROPDOWN_HH

#include "dropdown.hh"
#include "jetstream/render/sakura/components/button.hh"
#include "jetstream/render/sakura/components/stack.hh"

namespace Jetstream {

struct FlowgraphConfigWebUsbDropdownField : Sakura::Stack {
    using Config = FlowgraphConfigFieldConfig;

    Result update(Config config);
    Sakura::Component& component();

 protected:
    void layout(const Sakura::Context& ctx) override;

 private:
    Config config;
    FlowgraphConfigDropdownField dropdown;
    Sakura::Button authorizeButton;
    int observedGeneration = -1;
};

}  // namespace Jetstream

#endif
