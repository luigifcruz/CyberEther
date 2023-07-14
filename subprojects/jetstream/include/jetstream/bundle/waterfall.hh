#ifndef JETSTREAM_BUNDLE_WATERFALL_BASE_HH
#define JETSTREAM_BUNDLE_WATERFALL_BASE_HH

#include "jetstream/instance.hh"
#include "jetstream/modules/waterfall.hh"

namespace Jetstream::Bundle {

template<Device DeviceId>
class WaterfallUI {
 public:
    using Module = Waterfall<DeviceId>;

    Result init(Instance& instance,
                const typename Module::Config& config,
                const typename Module::Input& input) {
        module = instance.addBlock<Waterfall, DeviceId>(config, input);

        return Result::SUCCESS;
    }

    Result draw() {
        ImGui::Begin("Waterfall");

        auto [x, y] = ImGui::GetContentRegionAvail();
        auto scale = ImGui::GetIO().DisplayFramebufferScale;
        auto [width, height] = module->viewSize({
            static_cast<U64>(x*scale.x),
            static_cast<U64>(y*scale.y)
        });
        ImGui::Image(module->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));

        if (ImGui::IsItemHovered() && ImGui::IsAnyMouseDown()) {
            if (position == 0) {
                position = (getRelativeMousePos().x / module->zoom()) + module->offset();
            }
            module->offset(position - (getRelativeMousePos().x / module->zoom()));
        } else {
            position = 0;
        }

        ImGui::End();

        return Result::SUCCESS;       
    }

    Result drawControl() {
        auto interpolate = module->interpolate();
        if (ImGui::Checkbox("Interpolate Waterfall", &interpolate)) {
            module->interpolate(interpolate);
        }

        auto zoom = module->zoom();
        if (ImGui::DragFloat("Waterfall Zoom", &zoom, 0.01, 1.0, 5.0, "%f", 0)) {
            module->zoom(zoom);
        }
            
        return Result::SUCCESS;       
    }

    Result drawInfo() {
            
        return Result::SUCCESS;       
    }

    constexpr Module& get() {
        return *module;
    }

 private:
    std::shared_ptr<Module> module;
    I32 position;

    ImVec2 getRelativeMousePos() {
        ImVec2 mousePositionAbsolute = ImGui::GetMousePos();
        ImVec2 screenPositionAbsolute = ImGui::GetItemRectMin();
        return ImVec2(mousePositionAbsolute.x - screenPositionAbsolute.x,
                      mousePositionAbsolute.y - screenPositionAbsolute.y);
    }
};
    
}  // namespace Jetstream::Bundle

#endif
