#ifndef JETSTREAM_BUNDLE_LINEPLOT_BASE_HH
#define JETSTREAM_BUNDLE_LINEPLOT_BASE_HH

#include "jetstream/instance.hh"
#include "jetstream/modules/lineplot.hh"

namespace Jetstream::Bundle {

template<Device DeviceId>
class LineplotUI {
 public:
    using Module = Lineplot<DeviceId>;

    Result init(Instance& instance,
                const typename Module::Config& config,
                const typename Module::Input& input) {
        module = instance.addBlock<Lineplot, DeviceId>(config, input);

        return Result::SUCCESS;
    }

    Result draw() {
        ImGui::Begin("Lineplot");
        
        auto [x, y] = ImGui::GetContentRegionAvail();
        auto scale = ImGui::GetIO().DisplayFramebufferScale;
        auto [width, height] = module->viewSize({
            static_cast<U64>(x*scale.x),
            static_cast<U64>(y*scale.y)
        });
        ImGui::Image(module->getTexture().raw(), ImVec2(width/scale.x, height/scale.y));

        ImGui::End();

        return Result::SUCCESS;       
    }

    Result drawControl() {
            
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
};

}  // namespace Jetstream::Bundle

#endif
